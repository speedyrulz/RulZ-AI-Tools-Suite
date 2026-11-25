// Per-model, per-mode defaults: the shared helpers (src/shared/modes.ts) and how SettingsStore keeps them
// (model-mode-defaults.json next to model-defaults.json).
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_LOAD_CONFIG, DEFAULT_SAMPLING } from '@shared/defaults'
import {
  adoptSampling,
  APP_MODES,
  builtInSampling,
  chatPrompt,
  leftoverSampling,
  matchesModeLoad,
  modeDefaultsEmpty,
  modeLoad,
  modePromptExtra,
  modeSampling,
  reloadFor,
  sameLoad,
  withPromptExtra,
  type ModeDefaults,
  type ModelModeDefaults
} from '@shared/modes'
import { FICTION_SAMPLING } from '@shared/novelPrompts'
import { reSampling } from '@shared/re'
import type { LoadConfig } from '@shared/types'
import { parseModeDefaults } from '../src/main/settings/schema'
import { SettingsStore } from '../src/main/settings/store'

const dirs: string[] = []
const tmp = (): string => {
  const d = mkdtempSync(join(tmpdir(), 'rulz-modes-'))
  dirs.push(d)
  return d
}
afterEach(() => {
  vi.restoreAllMocks()
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true })
})

const load = (over: Partial<LoadConfig> = {}): LoadConfig => ({ ...DEFAULT_LOAD_CONFIG, ...over })
const entry = (over: Partial<ModeDefaults> = {}): ModeDefaults => ({ load: null, sampling: null, systemPrompt: '', ...over })

describe('mode helpers', () => {
  it('knows when an entry says nothing', () => {
    expect(modeDefaultsEmpty(undefined)).toBe(true)
    expect(modeDefaultsEmpty(entry())).toBe(true)
    expect(modeDefaultsEmpty(entry({ systemPrompt: '  \n ' }))).toBe(true)
    expect(modeDefaultsEmpty(entry({ systemPrompt: 'x' }))).toBe(false)
    expect(modeDefaultsEmpty(entry({ load: load() }))).toBe(false)
    expect(modeDefaultsEmpty(entry({ sampling: DEFAULT_SAMPLING }))).toBe(false)
  })

  it('takes the mode\'s load settings, else the model\'s own', () => {
    const base = load({ contextLength: 8192 })
    const re = load({ contextLength: 65536 })
    expect(modeLoad(base, { re: entry({ load: re }) }, 're')).toEqual(re)
    expect(modeLoad(base, { re: entry({ load: re }) }, 'chat')).toBe(base)
    expect(modeLoad(base, { re: entry({ sampling: DEFAULT_SAMPLING }) }, 're')).toBe(base)
    expect(modeLoad(base, undefined, 're')).toBe(base)
    expect(modeLoad(base, { re: entry({ load: re }) }, undefined)).toBe(base)
  })

  it('gives each mode its built-in sampling unless the model sets its own, always as a copy', () => {
    expect(modeSampling('chat', undefined, DEFAULT_SAMPLING)).toEqual(DEFAULT_SAMPLING)
    expect(modeSampling('re', undefined, DEFAULT_SAMPLING)).toEqual(reSampling(DEFAULT_SAMPLING))
    expect(modeSampling('adventure', undefined, DEFAULT_SAMPLING)).toEqual(DEFAULT_SAMPLING)
    expect(modeSampling('writing', undefined, DEFAULT_SAMPLING)).toEqual(FICTION_SAMPLING)
    const own = { ...DEFAULT_SAMPLING, topK: 9, jsonSchema: '{"a":1}' }
    const got = modeSampling('writing', { writing: entry({ sampling: own }) }, DEFAULT_SAMPLING)
    expect(got).toEqual({ ...own, jsonSchema: '' })
    got.stop.push('X')
    got.avoidWords.push('Y')
    expect(own.stop).toEqual(DEFAULT_SAMPLING.stop)
    expect(FICTION_SAMPLING.avoidWords).not.toContain('Y')
    expect(modeSampling('adventure', undefined, DEFAULT_SAMPLING).stop).not.toContain('X')
  })

  it('picks the chat prompt and the extra texts', () => {
    expect(chatPrompt(undefined, 'app')).toBe('app')
    expect(chatPrompt({ chat: entry({ systemPrompt: '   ' }) }, 'app')).toBe('app')
    expect(chatPrompt({ chat: entry({ systemPrompt: 'mine' }) }, 'app')).toBe('mine')
    expect(modePromptExtra({ re: entry({ systemPrompt: '  more  ' }) }, 're')).toBe('more')
    expect(modePromptExtra({ re: entry({ systemPrompt: 'more' }) }, 'writing')).toBe('')
    expect(modePromptExtra(undefined, 'adventure')).toBe('')
    expect(withPromptExtra('built-in', ' extra ')).toBe('built-in\n\nextra')
    expect(withPromptExtra('built-in', '  ')).toBe('built-in')
    expect(withPromptExtra('built-in', undefined)).toBe('built-in')
  })

  it('compares load settings whatever the order of their fields', () => {
    const a = load({ contextLength: 4096 })
    const reordered = Object.fromEntries(Object.entries(a).reverse()) as unknown as LoadConfig
    expect(sameLoad(a, reordered)).toBe(true)
    expect(sameLoad(a, load({ contextLength: 2048 }))).toBe(false)
    expect(sameLoad(load({ adapters: [{ id: 'a', scale: 1 }] }), load({ adapters: [{ id: 'a', scale: 0.5 }] }))).toBe(false)
  })

  it("always takes the API server sampling from the model's default load settings", () => {
    const base = load({ contextLength: 8192, sampling: { ...DEFAULT_LOAD_CONFIG.sampling, temperature: 0.7 } })
    const re = load({ contextLength: 65536, sampling: { ...DEFAULT_LOAD_CONFIG.sampling, temperature: 1, topK: 0 } })
    const got = modeLoad(base, { re: entry({ load: re }) }, 're')
    expect(got.contextLength).toBe(65536)
    expect(got.sampling).toBe(base.sampling)
    expect(re.sampling.temperature).toBe(1)
  })

  describe("leftover API sampling in a mode's load settings", () => {
    const base = load()
    const mine = (over: Partial<LoadConfig['sampling']>): LoadConfig => load({ sampling: { ...DEFAULT_LOAD_CONFIG.sampling, ...over } })

    it('is nothing when the mode has no load, or its sampling is empty or the same as the default one', () => {
      expect(leftoverSampling(null, base)).toBeNull()
      expect(leftoverSampling(undefined, base)).toBeNull()
      expect(leftoverSampling(mine({}), base)).toBeNull()
      expect(leftoverSampling(mine({ override: true }), base)).toBeNull()
      const same = mine({ temperature: 0.5 })
      expect(leftoverSampling(same, same)).toBeNull()
    })

    it('lists every value that is set when one differs, never override', () => {
      const own = mine({ temperature: 1, topK: 0, drySequenceBreakers: ['\n'], override: true })
      expect(leftoverSampling(own, base)).toEqual({ temperature: 1, topK: 0, drySequenceBreakers: ['\n'] })
      const other = load({ sampling: { ...base.sampling, topP: 1 } })
      expect(leftoverSampling(mine({ topP: 1, topK: 5 }), other)).toEqual({ topP: 1, topK: 5 })
    })

    it("adopted values go over the mode's built-in sampling", () => {
      const got = adoptSampling('re', DEFAULT_SAMPLING, { temperature: 1, topK: 0 })
      expect(got).toEqual({ ...builtInSampling('re', DEFAULT_SAMPLING), temperature: 1, topK: 0, jsonSchema: '' })
      expect(adoptSampling('writing', DEFAULT_SAMPLING, {}).temperature).toBe(FICTION_SAMPLING.temperature)
    })
  })

  describe('reloadFor', () => {
    const base = load({ contextLength: 8192 })
    const big = load({ contextLength: 65536 })
    const huge = load({ contextLength: 131072 })
    const modes: ModelModeDefaults = { re: entry({ load: big }), writing: entry({ load: huge }) }

    it('offers nothing when the model runs with what the mode wants', () => {
      expect(reloadFor(big, base, modes, 're')).toBeNull()
      expect(reloadFor(base, base, modes, 'chat')).toBeNull()
    })

    it('offers the mode\'s own settings when the model runs with others', () => {
      expect(reloadFor(base, base, modes, 're')).toEqual(big)
      expect(reloadFor(big, base, modes, 'writing')).toEqual(huge)
    })

    it('offers the model\'s default settings in a mode without load settings, when it runs with another mode\'s', () => {
      expect(reloadFor(big, base, modes, 'chat')).toEqual(base)
      expect(reloadFor(huge, base, modes, 'adventure')).toEqual(base)
    })

    it('leaves a model loaded by hand with settings of its own alone in a mode without any', () => {
      expect(reloadFor(load({ contextLength: 1234 }), base, modes, 'chat')).toBeNull()
    })

    it('offers nothing for a model with no mode settings', () => {
      expect(reloadFor(big, base, undefined, 're')).toBeNull()
      expect(reloadFor(big, base, { re: entry({ sampling: DEFAULT_SAMPLING }) }, 're')).toBeNull()
    })

    it("does not offer a reload because of leftover API sampling in a mode's stored load", () => {
      const left = { ...big, sampling: { ...big.sampling, temperature: 1, topK: 0 } }
      const withLeft: ModelModeDefaults = { re: entry({ load: left }) }
      // Loaded now: runs with the mode's load plus the default API sampling.
      expect(reloadFor(modeLoad(base, withLeft, 're'), base, withLeft, 're')).toBeNull()
      // Loaded before: runs with the stored load as it is, leftover included.
      expect(reloadFor(left, base, withLeft, 're')).toBeNull()
      expect(matchesModeLoad(left, base, withLeft, 're')).toBe(true)
      expect(reloadFor(left, base, withLeft, 'chat')).toEqual(base)
      expect(reloadFor(base, base, withLeft, 're')).toEqual(modeLoad(base, withLeft, 're'))
    })

    it("a stored load that only differs in sampling still counts as the mode's own", () => {
      const only: ModelModeDefaults = { re: entry({ load: { ...base, sampling: { ...base.sampling, temperature: 1 } } }) }
      expect(reloadFor(base, base, only, 're')).toBeNull()
      expect(reloadFor(big, base, only, 're')).toEqual(base)
    })

    it('does not mind the order of the fields of a running config', () => {
      const reordered = Object.fromEntries(Object.entries(big).reverse()) as unknown as LoadConfig
      expect(reloadFor(reordered, base, modes, 're')).toBeNull()
    })
  })
})

describe('parseModeDefaults', () => {
  it('keeps good entries and drops the rest, never losing the others for one bad entry', () => {
    const parsed = parseModeDefaults({
      good: { re: { load: { contextLength: 4096 }, sampling: null, systemPrompt: 'x' }, bogus: { systemPrompt: 'y' }, chat: { load: null, sampling: null, systemPrompt: '   ' } },
      broken: 'text',
      alsoBroken: [1, 2],
      nothing: { re: {}, writing: null },
      second: { adventure: { sampling: { topK: 'many', temperature: 1.5, jsonSchema: 'schema' } } }
    })
    expect(Object.keys(parsed).sort()).toEqual(['good', 'second'])
    expect(Object.keys(parsed.good)).toEqual(['re'])
    expect(parsed.good.re).toMatchObject({ load: { contextLength: 4096 }, sampling: null, systemPrompt: 'x' })
    // a bad field falls back on its own; a JSON schema never stays
    expect(parsed.second.adventure?.sampling).toMatchObject({ temperature: 1.5, topK: DEFAULT_SAMPLING.topK, jsonSchema: '' })
    expect(parseModeDefaults(null)).toEqual({})
    expect(parseModeDefaults([])).toEqual({})
  })

  it('treats ids like __proto__ as ordinary models', () => {
    const parsed = parseModeDefaults(JSON.parse('{"__proto__":{"re":{"systemPrompt":"p"}},"constructor":{"chat":{"systemPrompt":"c"}}}'))
    expect(Object.keys(parsed).sort()).toEqual(['__proto__', 'constructor'])
    expect(Object.getPrototypeOf(parsed)).toBe(Object.prototype)
    expect(({} as Record<string, unknown>).re).toBeUndefined()
  })
})

describe('SettingsStore mode defaults', () => {
  async function open(dir = tmp()): Promise<{ dir: string; store: SettingsStore }> {
    const store = new SettingsStore(dir)
    await store.init()
    return { dir, store }
  }
  const file = (dir: string): string => join(dir, 'model-mode-defaults.json')
  const saved = (dir: string): unknown => JSON.parse(readFileSync(file(dir), 'utf8'))

  it('starts empty and writes nothing until something is set', async () => {
    const { dir, store } = await open()
    expect(store.allModeDefaults()).toEqual({})
    expect(store.modeDefaults('m')).toEqual({})
    expect(existsSync(file(dir))).toBe(false)
  })

  it('saves a mode, reads it back after a restart, and leaves model-defaults.json alone', async () => {
    const { dir, store } = await open()
    await store.setModelDefaults('m', load({ contextLength: 2048 }))
    const before = readFileSync(join(dir, 'model-defaults.json'), 'utf8')
    await store.setModeDefaults('m', 're', entry({ load: load({ contextLength: 65536 }), sampling: { ...DEFAULT_SAMPLING, topK: 7, jsonSchema: 'x' }, systemPrompt: 'Be brief.' }))
    await store.setModeDefaults('m', 'chat', entry({ systemPrompt: 'Chatty.' }))
    expect(readFileSync(join(dir, 'model-defaults.json'), 'utf8')).toBe(before)
    const { store: again } = await open(dir)
    expect(Object.keys(again.modeDefaults('m')).sort()).toEqual(['chat', 're'])
    expect(again.modeDefaults('m').re).toMatchObject({ load: { contextLength: 65536 }, systemPrompt: 'Be brief.', sampling: { topK: 7, jsonSchema: '' } })
    expect(again.allModeDefaults()).toEqual(store.allModeDefaults())
  })

  it('does not hand out its own memory', async () => {
    const { store } = await open()
    await store.setModeDefaults('m', 'writing', entry({ sampling: DEFAULT_SAMPLING, systemPrompt: 'x' }))
    store.modeDefaults('m').writing!.sampling!.stop.push('X')
    store.allModeDefaults().m.writing!.systemPrompt = 'changed'
    store.loadFor('m', 'writing').adapters.push({ id: 'x', scale: 1 })
    expect(store.modeDefaults('m').writing!.sampling!.stop).toEqual(DEFAULT_SAMPLING.stop)
    expect(store.allModeDefaults().m.writing!.systemPrompt).toBe('x')
  })

  it('checks what it is given: unknown mode throws, junk is cleaned, an empty entry or null forgets the mode', async () => {
    const { dir, store } = await open()
    await expect(store.setModeDefaults('m', 'nope' as never, entry({ systemPrompt: 'x' }))).rejects.toThrow(/Unknown mode/)
    expect(existsSync(file(dir))).toBe(false)
    await store.setModeDefaults('m', 're', { load: { contextLength: -5 }, sampling: { topK: 'x' }, systemPrompt: 5 } as never)
    // the load settings and sampling were junk but present, so they are kept as defaults; the prompt is not text
    expect(store.modeDefaults('m').re).toMatchObject({ load: { contextLength: DEFAULT_LOAD_CONFIG.contextLength }, systemPrompt: '' })
    await store.setModeDefaults('m', 're', entry({ systemPrompt: '  ' }))
    expect(store.modeDefaults('m')).toEqual({})
    expect(saved(dir)).toEqual({})
    await store.setModeDefaults('m', 'chat', entry({ systemPrompt: 'a' }))
    await store.setModeDefaults('m', 'chat', null)
    expect(store.allModeDefaults()).toEqual({})
  })

  it('removes a model once its last mode goes, and keeps the other modes and models', async () => {
    const { dir, store } = await open()
    await store.setModeDefaults('m', 'chat', entry({ systemPrompt: 'a' }))
    await store.setModeDefaults('m', 're', entry({ systemPrompt: 'b' }))
    await store.setModeDefaults('n', 're', entry({ systemPrompt: 'c' }))
    await store.setModeDefaults('m', 'chat', null)
    expect(Object.keys(store.allModeDefaults()).sort()).toEqual(['m', 'n'])
    await store.setModeDefaults('m', 're', null)
    expect(Object.keys(saved(dir) as object)).toEqual(['n'])
  })

  it('loses nothing when one stored entry is bad or the file has unknown parts', async () => {
    const dir = tmp()
    writeFileSync(
      file(dir),
      JSON.stringify({ a: { re: { systemPrompt: 'keep' }, future: { systemPrompt: 'later' } }, b: 'junk', c: { chat: { load: 'bad', sampling: 5, systemPrompt: 'ok' } } })
    )
    const { store } = await open(dir)
    expect(Object.keys(store.allModeDefaults()).sort()).toEqual(['a', 'c'])
    expect(store.modeDefaults('a').re?.systemPrompt).toBe('keep')
    expect(store.modeDefaults('c').chat).toEqual({ load: null, sampling: null, systemPrompt: 'ok' })
  })

  it('moves a file that is not JSON aside and starts empty', async () => {
    const dir = tmp()
    writeFileSync(file(dir), '{ not json')
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const { store } = await open(dir)
    expect(store.allModeDefaults()).toEqual({})
    expect(warn).toHaveBeenCalled()
    expect(readdirSync(dir).some((f) => f.startsWith('model-mode-defaults.json.corrupt-'))).toBe(true)
  })

  it('never overwrites a file it could not read, and keeps memory as it was', async () => {
    const dir = tmp()
    // A folder where the file should be: it exists but cannot be read as a file.
    mkdirSync(file(dir))
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const { store } = await open(dir)
    await expect(store.setModeDefaults('m', 're', entry({ systemPrompt: 'x' }))).rejects.toThrow(/wasn't overwritten/)
    expect(store.allModeDefaults()).toEqual({})
    expect(readdirSync(file(dir))).toEqual([])
    // the other files still save
    await store.setModelDefaults('m', load({ contextLength: 1024 }))
    expect(JSON.parse(readFileSync(join(dir, 'model-defaults.json'), 'utf8')).m.contextLength).toBe(1024)
  })

  describe('loadFor', () => {
    it('falls back to the model\'s defaults, then to the app\'s', async () => {
      const { store } = await open()
      expect(store.loadFor('m')).toEqual(store.get().defaultLoadConfig)
      expect(store.loadFor('m', 're')).toEqual(store.get().defaultLoadConfig)
      await store.setModelDefaults('m', load({ contextLength: 2048 }))
      expect(store.loadFor('m', 're').contextLength).toBe(2048)
      await store.setModeDefaults('m', 're', entry({ load: load({ contextLength: 65536 }) }))
      expect(store.loadFor('m', 're').contextLength).toBe(65536)
      expect(store.loadFor('m', 'chat').contextLength).toBe(2048)
      expect(store.loadFor('m').contextLength).toBe(2048)
      await store.setModeDefaults('m', 'writing', entry({ systemPrompt: 'only a prompt' }))
      expect(store.loadFor('m', 'writing').contextLength).toBe(2048)
    })

    it('does not read an id like __proto__ or constructor as stored settings', async () => {
      const { store } = await open()
      for (const id of ['__proto__', 'constructor', 'toString']) {
        expect(store.loadFor(id, 're')).toEqual(store.get().defaultLoadConfig)
        expect(store.getModelDefaults(id)).toEqual({ config: store.get().defaultLoadConfig, custom: false })
        expect(store.modeDefaults(id)).toEqual({})
      }
    })

    it('stores settings under such an id like any other', async () => {
      const { dir, store } = await open()
      await store.setModeDefaults('__proto__', 're', entry({ load: load({ contextLength: 999 }) }))
      expect(store.loadFor('__proto__', 're').contextLength).toBe(999)
      expect(store.loadFor('other', 're').contextLength).toBe(DEFAULT_LOAD_CONFIG.contextLength)
      expect(Object.getPrototypeOf(store.allModeDefaults())).toBe(Object.prototype)
      const { store: again } = await open(dir)
      expect(again.loadFor('__proto__', 're').contextLength).toBe(999)
    })
  })

  it('saves that overlap all stay, and a forget after them wins', async () => {
    const dir = tmp()
    const store = new SettingsStore(dir)
    await store.init()
    const entry = (systemPrompt: string): ModeDefaults => ({ load: null, sampling: null, systemPrompt })
    await Promise.all([store.setModeDefaults('a', 'chat', entry('1')), store.setModeDefaults('a', 're', entry('2')), store.setModeDefaults('b', 'writing', entry('3'))])
    expect(Object.keys(store.modeDefaults('a')).sort()).toEqual(['chat', 're'])
    expect(store.modeDefaults('b').writing?.systemPrompt).toBe('3')
    await Promise.all([store.setModeDefaults('c', 'chat', entry('4')), store.forgetModel('c')])
    const again = new SettingsStore(dir)
    await again.init()
    expect(Object.keys(again.allModeDefaults()).sort()).toEqual(['a', 'b'])
  })

  describe('forgetModel', () => {
    it('clears the default load settings and every mode of that model only', async () => {
      const { dir, store } = await open()
      await store.setModelDefaults('m', load({ contextLength: 2048 }))
      await store.setModelDefaults('n', load({ contextLength: 4096 }))
      await store.setModeDefaults('m', 're', entry({ systemPrompt: 'a' }))
      await store.setModeDefaults('m', 'chat', entry({ systemPrompt: 'b' }))
      await store.setModeDefaults('n', 're', entry({ systemPrompt: 'c' }))
      await store.forgetModel('m')
      expect(store.getModelDefaults('m').custom).toBe(false)
      expect(store.modeDefaults('m')).toEqual({})
      expect(store.getModelDefaults('n').custom).toBe(true)
      expect(store.modeDefaults('n').re?.systemPrompt).toBe('c')
      const { store: again } = await open(dir)
      expect(again.getModelDefaults('m').custom).toBe(false)
      expect(Object.keys(again.allModeDefaults())).toEqual(['n'])
    })

    it('writes only the files that change', async () => {
      const { dir, store } = await open()
      await store.forgetModel('ghost')
      expect(existsSync(file(dir))).toBe(false)
      expect(existsSync(join(dir, 'model-defaults.json'))).toBe(false)
      await store.setModeDefaults('m', 're', entry({ systemPrompt: 'a' }))
      await store.forgetModel('m')
      // m had no default load settings of its own, so that file was never written
      expect(existsSync(join(dir, 'model-defaults.json'))).toBe(false)
      expect(saved(dir)).toEqual({})
      await store.setModelDefaults('only-load', load({ contextLength: 1 }))
      const before = readFileSync(file(dir), 'utf8')
      await store.forgetModel('only-load')
      expect(readFileSync(file(dir), 'utf8')).toBe(before)
    })

    it('does not touch a mode file that could not be read when only the load defaults change', async () => {
      const dir = tmp()
      mkdirSync(file(dir))
      vi.spyOn(console, 'error').mockImplementation(() => undefined)
      const { store } = await open(dir)
      await store.setModelDefaults('m', load({ contextLength: 2048 }))
      await expect(store.forgetModel('m')).resolves.toBeUndefined()
      expect(store.getModelDefaults('m').custom).toBe(false)
    })
  })

  it('knows every mode', () => {
    expect(APP_MODES).toEqual(['chat', 're', 'adventure', 'writing'])
  })
})
