// OpenRouter models in the catalog, the writing adapters and the phone, and a guard against "this PC" tests that forget them.
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative, sep } from 'node:path'
import { describe, expect, it, vi } from 'vitest'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import type { LoadedInstance, ModelEntry } from '@shared/types'
import type { InstanceManager } from '../src/main/engine/instances'
import { createModelCatalog } from '../src/main/models/catalog'
import { asPhone, chatAdapter, modelsAdapter } from '../src/main/writing/adapters'
import { CLOUD_RELOAD, PhoneModes, type PhoneModesDeps } from '../src/main/writing/modeInfo'
import { WritingRuns } from '../src/main/writing/runs'
import { cloudEntry, cloudInstance, localEntry } from './helpers/cloudFakes'

describe('the model catalog with OpenRouter as a third source', () => {
  const local = [localEntry('A/a.gguf'), localEntry('B/b.gguf')]
  const library = { list: () => local, get: (id: string) => local.find((m) => m.id === id) }
  const rentals = (): ModelEntry[] => [{ ...localEntry('vast/big'), remote: true }]

  it('lists library, rentals, then cloud; cloud entries keep `cloud` and never get `remote`', () => {
    const tagged = { ...cloudEntry('x/one'), remote: true } as ModelEntry
    const catalog = createModelCatalog(library, rentals, () => [tagged, cloudEntry('y/two')])
    expect(catalog.list().map((m) => [m.id, !!m.remote, !!m.cloud])).toEqual([
      ['A/a.gguf', false, false],
      ['B/b.gguf', false, false],
      ['vast/big', true, false],
      ['openrouter/x/one', false, true],
      ['openrouter/y/two', false, true]
    ])
    expect(catalog.get('openrouter/x/one')?.cloud?.slug).toBe('x/one')
    expect(catalog.get('openrouter/x/one')?.remote).toBeUndefined()
    expect(catalog.get('vast/big')?.remote).toBe(true)
    expect(catalog.get('nope')).toBeUndefined()
  })

  it('lets the library win an id clash, and works without a cloud source', () => {
    const catalog = createModelCatalog(library, rentals, () => [cloudEntry('x/one', {}), { ...cloudEntry('q'), id: 'A/a.gguf' }])
    expect(catalog.list().filter((m) => m.id === 'A/a.gguf')).toHaveLength(1)
    expect(catalog.get('A/a.gguf')?.cloud).toBeUndefined()
    expect(createModelCatalog(library, rentals).list()).toHaveLength(3)
  })
})

describe('modelsAdapter and chatAdapter with OpenRouter models', () => {
  const ALLOWED = cloudEntry('a/ok', { allowApps: true })
  const OFF = cloudEntry('b/off', { allowApps: false })
  const LOCAL = localEntry('local')
  const library = { list: () => [LOCAL, ALLOWED, OFF] }
  const allowed = (id: string): boolean => id === ALLOWED.id
  const manager = (running: LoadedInstance[], ensure = vi.fn(async (id: string) => ({ instance: cloudInstance(id.replace('openrouter/', '')), started: true }))): InstanceManager =>
    ({ readyFor: (id: string) => running.find((i) => i.modelId === id && i.status === 'ready'), list: () => running, ensure }) as unknown as InstanceManager

  it('lists every cloud model for the window and only the ticked ones for a phone, with their free flag', () => {
    const win = modelsAdapter(library, manager([]))
    expect(win.list().map((m) => m.id)).toEqual(['local', ALLOWED.id, OFF.id])
    expect(win.list()[1]).toMatchObject({ cloud: { free: false } })
    const phone = modelsAdapter(library, manager([]), { by: 'phone', cloudAllowed: allowed })
    expect(phone.list().map((m) => m.id)).toEqual(['local', ALLOWED.id])
    // The window's adapter acts as a phone inside asPhone (adventure jobs a phone starts).
    const shared = modelsAdapter(library, manager([]), { cloudAllowed: allowed })
    expect(shared.list().map((m) => m.id)).toEqual(['local', ALLOWED.id, OFF.id])
    expect(asPhone(() => shared.list().map((m) => m.id))).toEqual(['local', ALLOWED.id])
  })

  it('ready(null) is never a cloud model, even the newest one; ready(id) works for a chosen one', () => {
    const running = [{ ...cloudInstance('a/ok'), startedAt: 9 }, { ...cloudInstance('local'), modelId: 'local', cloud: undefined, startedAt: 1 }] as LoadedInstance[]
    const m = modelsAdapter(library, manager(running))
    expect(m.ready(null)?.modelId).toBe('local')
    expect(modelsAdapter(library, manager([running[0]])).ready(null)).toBeUndefined()
    expect(m.ready('openrouter/a/ok')).toMatchObject({ modelId: 'openrouter/a/ok', cloud: { free: false } })
  })

  it('ensure connects an explicitly chosen cloud model for the window, and refuses a not-ticked one for a phone with a sentence', async () => {
    const ensure = vi.fn(async (id: string) => ({ instance: cloudInstance(id.replace('openrouter/', '')), started: true }))
    const win = modelsAdapter(library, manager([], ensure))
    await expect(win.ensure(OFF.id)).resolves.toMatchObject({ modelId: OFF.id })
    const phone = modelsAdapter(library, manager([], ensure), { by: 'phone', cloudAllowed: allowed })
    await expect(phone.ensure(OFF.id)).rejects.toThrow(/not ticked "Allow other apps"/)
    await expect(phone.ensure(ALLOWED.id)).resolves.toMatchObject({ modelId: ALLOWED.id })
    // A loaded not-ticked model is not handed to a phone either.
    const loaded = modelsAdapter(library, manager([cloudInstance('b/off')]), { by: 'phone', cloudAllowed: allowed })
    expect(loaded.ready(OFF.id)).toBeUndefined()
  })

  it('chatAdapter says who asks on every call, asPhone included', async () => {
    const chat = { countText: vi.fn(async () => 1), countTokens: vi.fn(async () => 1), start: vi.fn(async () => undefined), abort: vi.fn() }
    const tap = { subscribe: vi.fn() }
    const adapter = chatAdapter(chat as never, tap as never)
    await adapter.countText('m', 't')
    await adapter.countTokens('m', [])
    await adapter.start({} as never)
    expect(chat.countText).toHaveBeenLastCalledWith('m', 't', undefined, 'window')
    expect(chat.countTokens).toHaveBeenLastCalledWith('m', [], null, undefined, 'window')
    expect(chat.start).toHaveBeenLastCalledWith({}, 'window')
    await asPhone(async () => {
      await adapter.countText('m', 't')
      await adapter.start({} as never)
      // Still the phone after an await: the job a phone starts goes on in the background.
      await new Promise((r) => setTimeout(r, 1))
      await adapter.countTokens('m', [])
    })
    expect(chat.countText).toHaveBeenLastCalledWith('m', 't', undefined, 'phone')
    expect(chat.start).toHaveBeenLastCalledWith({}, 'phone')
    expect(chat.countTokens).toHaveBeenLastCalledWith('m', [], null, undefined, 'phone')
    const phone = chatAdapter(chat as never, tap as never, 'phone')
    await phone.start({} as never)
    expect(chat.start).toHaveBeenLastCalledWith({}, 'phone')
  })

  it('WritingRuns.models adds cloud: {free} for the phone', () => {
    const runs = new WritingRuns({ store: {} as never, chat: {} as never, models: modelsAdapter(library, manager([cloudInstance('a/ok')]), { by: 'phone', cloudAllowed: allowed }), emit: () => undefined })
    expect(runs.models()).toEqual([
      { id: 'local', name: 'local', loaded: false },
      { id: ALLOWED.id, name: 'a/ok', loaded: true, cloud: { free: false } }
    ])
  })
})

describe('PhoneModes with an OpenRouter model', () => {
  const ALLOWED = cloudEntry('a/ok', { allowApps: true })
  const OFF = cloudEntry('b/off', { allowApps: false })
  const deps = (running: LoadedInstance[]): PhoneModesDeps => ({
    getEntry: (id) => [ALLOWED, OFF, localEntry('local')].find((m) => m.id === id),
    cloudAllowed: (id) => id === ALLOWED.id,
    newestReady: () => null,
    readyFor: (id) => running.find((i) => i.modelId === id),
    modeDefaults: () => ({}),
    loadDefaults: () => DEFAULT_LOAD_CONFIG,
    reloadBlock: () => null,
    leased: () => false,
    reload: () => {
      throw new Error('a cloud model is never reloaded')
    }
  })

  it('shows the cloud flag and never offers a reload, for a ticked model', () => {
    // Settings that differ from what runs would normally offer a reload; for a cloud model there is none.
    const running = [cloudInstance('a/ok', { config: { ...DEFAULT_LOAD_CONFIG, contextLength: 1 } })]
    const info = new PhoneModes(deps(running)).info('writing', ALLOWED.id)
    expect(info).toMatchObject({ modelId: ALLOWED.id, name: 'a/ok', loaded: true, cloud: { free: false }, reload: null })
  })

  it('treats a not-ticked model as unknown, and refuses to reload a cloud model', () => {
    const modes = new PhoneModes(deps([cloudInstance('b/off')]))
    expect(modes.info('writing', OFF.id)).toMatchObject({ name: OFF.id, loaded: false })
    expect(modes.info('writing', OFF.id).cloud).toBeUndefined()
    expect(() => modes.reloadMode('writing', OFF.id)).toThrow(/not in the library/)
    expect(() => modes.reloadMode('writing', ALLOWED.id)).toThrow(CLOUD_RELOAD)
  })
})

// ---- "this PC" tests: a cloud instance has no host either ----

/** Every .ts file under a folder. */
function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name)
    return statSync(path).isDirectory() ? sources(path) : path.endsWith('.ts') ? [path] : []
  })
}

describe('no "this PC" test that forgets OpenRouter', () => {
  const root = join(__dirname, '..', 'src', 'main')
  /**
   * `!x.host` and `!x.remote` mean "runs on this PC" in the old code, and an OpenRouter model is neither: use
   * isLocalInstance / isLocalModel from @shared/cloud. Allowed places, with the reason:
   */
  const allowed: { file: string; why: string }[] = [
    // The rental service only ever sees rentals (src/main/remote is skipped below), so there is nothing to allow there.
    { file: 'index.ts', why: 'prepareHosts: a cloud model was refused or handled before the rental branch, so `!model.remote` there means a library model' },
    { file: 'server/apiServer.ts', why: 'visibleModels: the `m.cloud ?` branch comes first, so `!m.remote` is only asked of library and rental models' },
    { file: 'engine/instances.ts', why: 'resolveHost / startLoad: a cloud model is handled before these lines, so `!model.remote` there means a library model' }
  ]
  const pattern = /(?<!!)!\s*[\w.?]+\.(host|remote)\b(?![.\s]*[=!:.])/g

  it('finds none in src/main (except main/remote) beyond the allowed ones', () => {
    const hits: string[] = []
    for (const file of sources(root)) {
      const rel = relative(root, file).split(sep).join('/')
      if (rel.startsWith('remote/') || rel.startsWith('cloud/')) continue
      const lines = readFileSync(file, 'utf8').split(/\r?\n/)
      lines.forEach((line, i) => {
        if (/^\s*(\/\/|\*|\/\*)/.test(line)) return
        for (const m of line.matchAll(pattern)) {
          // `!model.cloud`-guarded lines are fine: the cloud case was handled first.
          if (allowed.some((a) => a.file === rel)) continue
          hits.push(`${rel}:${i + 1}: ${m[0]}`)
        }
      })
    }
    expect(hits, 'use isLocalInstance / isLocalModel from @shared/cloud where "this PC" is meant').toEqual([])
  })

  it('the allowlist names files that exist', () => {
    for (const a of allowed) expect(() => statSync(join(root, a.file))).not.toThrow()
  })
})
