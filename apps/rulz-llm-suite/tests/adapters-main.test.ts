import type { ChildProcess } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { existsSync, mkdirSync, mkdtempSync, writeFileSync } from 'node:fs'
import http from 'node:http'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { requestLora } from '@shared/adapters'
import { DEFAULT_LOAD_CONFIG, DEFAULT_SAMPLING } from '@shared/defaults'
import type { AdapterConvertRequest, AdapterEntry, GgufInfo, LoadConfig, LoadedInstance, MemoryEstimate, ModelEntry } from '@shared/types'
import { buildRequestBody } from '../src/main/chat/service'
import { buildServerArgs } from '../src/main/engine/args'
import type { EngineService } from '../src/main/engine/engine'
import { InstanceManager, resolveAdapters, type InstanceManagerDeps } from '../src/main/engine/instances'
import { AdapterConverter, buildConvertArgs, findConverterScript, sanitizeOutName } from '../src/main/models/convert'
import { estimateMemory } from '../src/main/models/estimate'
import { toGgufInfo } from '../src/main/models/gguf'
import { classifyFile, groupAdapters, groupModels, type ScannedFile } from '../src/main/models/library'
import { BatchStore, cleanRun } from '../src/main/re/batches'
import { ApiServer } from '../src/main/server/apiServer'
import { adapterChoiceList, adaptersFromRemote, parseRemoteLoadPatch } from '../src/main/server/helpers'
import { parseChatAdapters, parseLoadConfig } from '../src/main/settings/schema'

const root = 'C:\\LLM'
const adapterInfo = (extra: Partial<GgufInfo> = {}): GgufInfo => ({
  architecture: 'llama',
  type: 'adapter',
  adapterType: 'lora',
  loraAlpha: 32,
  baseModelName: 'Llama 3 8B',
  hasChatTemplate: false,
  ...extra
})
const scanned = (rel: string, size: number, info: GgufInfo | null): ScannedFile => ({ path: join(root, rel), sizeBytes: size, mtimeMs: 5, info })

const adapterEntry = (id: string, extra: Partial<AdapterEntry> = {}): AdapterEntry => ({
  id,
  name: id.replace(/\.gguf$/, ''),
  path: join(root, ...id.split('/')),
  sizeBytes: 100e6,
  architecture: 'llama',
  info: adapterInfo(),
  modifiedAt: 1,
  ...extra
})

const llama: ModelEntry = {
  id: 'Llama/llama-3-8b.gguf',
  name: 'llama-3-8b',
  folder: 'Llama',
  path: 'C:\\LLM\\Llama\\llama-3-8b.gguf',
  files: ['C:\\LLM\\Llama\\llama-3-8b.gguf'],
  sizeBytes: 8e9,
  info: { architecture: 'llama', name: 'Llama 3 8B', blockCount: 32, headCountKv: 8, keyLength: 128, valueLength: 128, hasChatTemplate: true },
  modifiedAt: 0
}

describe('adapter scan', () => {
  const files = [
    scanned('Llama/llama-3-8b.gguf', 8e9, { ...adapterInfo(), type: undefined, name: 'Llama 3 8B', baseModelName: undefined }),
    scanned('Loras/style.gguf', 50e6, adapterInfo()),
    scanned('Loras/odd.gguf', 10e6, adapterInfo({ architecture: undefined as unknown as string })),
    scanned('Loras/broken.gguf', 5, null)
  ]

  it('classifies general.type adapter files and produces AdapterEntry records', () => {
    expect(classifyFile('style.gguf', adapterInfo())).toBe('adapter')
    const adapters = groupAdapters(root, files)
    expect(adapters.map((a) => a.id)).toEqual(['Loras/odd.gguf', 'Loras/style.gguf'])
    const style = adapters.find((a) => a.id === 'Loras/style.gguf')!
    expect(style).toMatchObject({ name: 'style', path: join(root, 'Loras', 'style.gguf'), sizeBytes: 50e6, architecture: 'llama', modifiedAt: 5 })
    expect(adapters.find((a) => a.id === 'Loras/odd.gguf')!.architecture).toBe('unknown')
  })

  it('never lists an adapter as a model', () => {
    const models = groupModels(root, files)
    expect(models.map((m) => m.id)).toEqual(['Llama/llama-3-8b.gguf', 'Loras/broken.gguf'])
  })
})

describe('GGUF adapter metadata', () => {
  it('reads type, alpha and the base model name', () => {
    const kv = {
      'general.architecture': 'qwen3',
      'general.type': 'adapter',
      'adapter.type': 'lora',
      'adapter.lora.alpha': 16,
      'general.base_model.0.name': 'Qwen3 8B',
      'general.basename': 'other'
    }
    expect(toGgufInfo(kv, 'a.gguf')).toMatchObject({ type: 'adapter', adapterType: 'lora', loraAlpha: 16, baseModelName: 'Qwen3 8B' })
  })
  it('falls back to general.basename and leaves absent fields out', () => {
    expect(toGgufInfo({ 'general.architecture': 'llama', 'general.basename': 'Llama-3' }, 'a.gguf').baseModelName).toBe('Llama-3')
    const plain = toGgufInfo({ 'general.architecture': 'llama' }, 'a.gguf')
    expect(plain.adapterType).toBeUndefined()
    expect(plain.loraAlpha).toBeUndefined()
    expect(plain.baseModelName).toBeUndefined()
  })
})

describe('buildServerArgs adapters', () => {
  const args = (adapters?: { path: string }[]): string[] =>
    buildServerArgs({ model: llama, config: DEFAULT_LOAD_CONFIG, port: 1, alias: 'x', adapters })

  it('passes one --lora per adapter, in order, and never --lora-scaled', () => {
    const a = args([{ path: 'C:\\LLM\\Loras\\b.gguf' }, { path: 'C:\\LLM\\Loras\\a.gguf' }])
    const pairs = a.flatMap((v, i) => (v === '--lora' ? [a[i + 1]] : []))
    expect(pairs).toEqual(['C:\\LLM\\Loras\\b.gguf', 'C:\\LLM\\Loras\\a.gguf'])
    expect(a).not.toContain('--lora-scaled')
  })
  it('adds nothing without adapters', () => {
    expect(args()).not.toContain('--lora')
    expect(args([])).not.toContain('--lora')
  })
})

describe('load config adapters', () => {
  it('cleans the list: bad entries and duplicates dropped, scales clamped, at most 16', () => {
    const cfg = parseLoadConfig({
      adapters: [{ id: 'a.gguf', scale: 9 }, { id: '', scale: 1 }, { scale: 1 }, null, 'x', { id: 'a.gguf', scale: 0.5 }, { id: 'b.gguf', scale: 'x' }, { id: 'c.gguf', scale: -9 }]
    })
    expect(cfg.adapters).toEqual([
      { id: 'a.gguf', scale: 4 },
      { id: 'b.gguf', scale: 1 },
      { id: 'c.gguf', scale: -4 }
    ])
    const many = parseLoadConfig({ adapters: Array.from({ length: 30 }, (_, i) => ({ id: `${i}.gguf`, scale: 1 })) })
    expect(many.adapters).toHaveLength(16)
  })
  it('falls back to [] on bad input and keeps the other fields', () => {
    expect(parseLoadConfig({ adapters: 'nope', parallel: 3 })).toMatchObject({ adapters: [], parallel: 3 })
    expect(parseLoadConfig({}).adapters).toEqual([])
    expect(parseLoadConfig(undefined).adapters).toEqual([])
  })
  it('gives a chat null or a cleaned list, never throwing', () => {
    expect(parseChatAdapters(undefined)).toBeNull()
    expect(parseChatAdapters('x')).toBeNull()
    expect(parseChatAdapters({ id: 'a' })).toBeNull()
    expect(parseChatAdapters([{ id: 'a.gguf', scale: 2 }, 7, { id: 'a.gguf', scale: 1 }])).toEqual([{ id: 'a.gguf', scale: 2 }])
    expect(parseChatAdapters([])).toEqual([])
  })
})

describe('request lora', () => {
  const loaded = [
    { id: 'a.gguf', name: 'a', path: 'C:\\a.gguf', scale: 0.5 },
    { id: 'b.gguf', name: 'b', path: 'C:\\b.gguf', scale: 1 }
  ]
  it('sets body.lora from the instance adapters and the chat strengths', () => {
    const body = buildRequestBody('m', [], DEFAULT_SAMPLING, false, requestLora(loaded, [{ id: 'b.gguf', scale: 0 }]))
    expect(body.lora).toEqual([
      { id: 0, scale: 0.5 },
      { id: 1, scale: 0 }
    ])
  })
  it('leaves the field out when the instance loaded no adapter', () => {
    expect(buildRequestBody('m', [], DEFAULT_SAMPLING, false, requestLora([], null))).not.toHaveProperty('lora')
    expect(buildRequestBody('m', [], DEFAULT_SAMPLING)).not.toHaveProperty('lora')
  })
})

describe('estimate adapterBytes', () => {
  const free = { vramBytes: 40e9, ramBytes: 60e9 }
  const at = (gpuLayers: LoadConfig['gpuLayers'], adapter: number): MemoryEstimate =>
    estimateMemory(llama, { ...DEFAULT_LOAD_CONFIG, contextLength: 4096, gpuLayers }, free, adapter)

  it('adds the adapter size to the total and, per offload, to the GPU share', () => {
    const none = at('all', 0)
    const all = at('all', 200e6)
    expect(all.adapterBytes).toBe(200e6)
    expect(none.adapterBytes).toBe(0)
    expect(all.totalBytes - none.totalBytes).toBeCloseTo(200e6)
    expect(all.gpuBytes - none.gpuBytes).toBeCloseTo(200e6)

    const part = at(16, 200e6)
    expect(part.gpuBytes - at(16, 0).gpuBytes).toBeCloseTo((200e6 * 16) / 33)
    expect(part.totalBytes - at(16, 0).totalBytes).toBeCloseTo(200e6)

    const zero = at(0, 200e6)
    expect(zero.gpuBytes).toBeCloseTo(at(0, 0).gpuBytes)
  })
  it('keeps the per-GPU split consistent with gpuBytes', () => {
    const gpus = [
      { index: 0, name: 'A', freeBytes: 20e9, totalBytes: 24e9 },
      { index: 1, name: 'B', freeBytes: 20e9, totalBytes: 24e9 }
    ]
    const est = estimateMemory(llama, { ...DEFAULT_LOAD_CONFIG, contextLength: 4096, gpuLayers: 'all' }, { ...free, gpus }, 300e6)
    const base = estimateMemory(llama, { ...DEFAULT_LOAD_CONFIG, contextLength: 4096, gpuLayers: 'all' }, { ...free, gpus }, 0)
    expect(est.perGpu.reduce((s, g) => s + g.bytes, 0)).toBeCloseTo(est.gpuBytes)
    expect(est.perGpu[0].bytes - base.perGpu[0].bytes).toBeCloseTo(150e6)
  })
  it('defaults to no adapters', () => {
    expect(estimateMemory(llama, DEFAULT_LOAD_CONFIG, free).adapterBytes).toBe(0)
  })
})

describe('load-config API adapters', () => {
  const freePort = async (): Promise<number> => {
    const s = http.createServer()
    await new Promise<void>((r) => s.listen(0, '127.0.0.1', () => r()))
    const port = (s.address() as { port: number }).port
    await new Promise<void>((r) => s.close(() => r()))
    return port
  }
  const adapters: AdapterEntry[] = [
    adapterEntry('Loras/style.gguf', { info: adapterInfo({ baseModelName: 'Llama 3 8B', loraAlpha: 16 }) }),
    adapterEntry('Loras/other.gguf', { info: adapterInfo({ baseModelName: 'Mistral 7B' }), sizeBytes: 5 }),
    adapterEntry('Loras/qwen.gguf', { architecture: 'qwen3' })
  ]

  it('lists choices, saves valid lists, refuses unknown and duplicate ids, clamps, and honours dry_run and reload', async () => {
    const saved: Record<string, LoadConfig> = {}
    const loads: string[] = []
    const ready = { instanceId: 'i1', modelId: llama.id, status: 'ready', config: DEFAULT_LOAD_CONFIG, contextLength: 8192 } as LoadedInstance
    const instances = {
      list: () => [ready],
      readyFor: () => ready,
      load: async (id: string) => {
        loads.push(id)
        return { ...ready, config: saved[id] }
      }
    } as unknown as InstanceManager
    const port = await freePort()
    const server = new ApiServer({
      instances,
      getModels: () => [llama],
      getAdapters: () => adapters,
      getSettings: () => ({ autoStart: false, port, bind: 'localhost', apiKey: '', cors: false, idleUnloadMinutes: 0, defaultModelId: '', unknownModelFallback: true }),
      getLoadDefaults: (id) => (saved[id] ? { config: saved[id], custom: true } : { config: DEFAULT_LOAD_CONFIG, custom: false }),
      setLoadDefaults: async (id, config) => void (saved[id] = config),
      estimate: () => null,
      gpus: () => [],
      onStatus: () => undefined,
      onRequest: () => undefined
    })
    await server.start()
    try {
      const url = `http://127.0.0.1:${port}/v1/models/${encodeURIComponent('llama-3-8b')}/load-config`
      const post = (body: unknown): Promise<Response> => fetch(url, { method: 'POST', body: JSON.stringify(body) })
      type Reply = { adapters: { id: string; scale: number }[]; adapter_choices: Record<string, unknown>[]; config: { adapters: unknown }; dry_run?: boolean }

      const got = (await (await fetch(url)).json()) as Reply
      expect(got.adapters).toEqual([])
      expect(got.adapter_choices).toEqual([
        { id: 'Loras/style.gguf', name: 'Loras/style', size_bytes: 100e6, base_model: 'Llama 3 8B', alpha: 16, base_match: 'match' },
        { id: 'Loras/other.gguf', name: 'Loras/other', size_bytes: 5, base_model: 'Mistral 7B', alpha: 32, base_match: 'mismatch' }
      ])

      const ok = await post({ config: { adapters: [{ id: 'Loras/style.gguf', scale: 9 }, { id: 'Loras/other.gguf', scale: 0.5 }] } })
      expect(ok.status).toBe(200)
      expect(saved[llama.id].adapters).toEqual([
        { id: 'Loras/style.gguf', scale: 4 },
        { id: 'Loras/other.gguf', scale: 0.5 }
      ])
      expect(((await ok.json()) as Reply).adapters).toHaveLength(2)
      expect(loads).toEqual([llama.id])

      const unknown = await post({ config: { adapters: [{ id: 'Loras/qwen.gguf', scale: 1 }] } })
      expect(unknown.status).toBe(400)
      expect(JSON.stringify(await unknown.json())).toContain('Invalid load setting adapters: Loras/qwen.gguf is not one of adapter_choices')
      expect((await post({ config: { adapters: [{ id: 'Loras/style.gguf', scale: 1 }, { id: 'Loras/style.gguf', scale: 2 }] } })).status).toBe(400)
      expect((await post({ config: { adapters: Array.from({ length: 17 }, () => ({ id: 'Loras/style.gguf', scale: 1 })) } })).status).toBe(400)
      expect((await post({ config: { adapters: [{ id: 'Loras/style.gguf' }] } })).status).toBe(400)
      expect(saved[llama.id].adapters).toHaveLength(2)

      const dry = (await (await post({ config: { adapters: [{ id: 'Loras/style.gguf', scale: 0.25 }] }, dry_run: true })).json()) as Reply
      expect(dry.dry_run).toBe(true)
      expect(dry.adapters).toEqual([{ id: 'Loras/style.gguf', scale: 0.25 }])
      expect(saved[llama.id].adapters).toHaveLength(2)
      expect(loads).toHaveLength(1)

      // Beside `config` works too; reload:false saves without restarting.
      expect((await post({ adapters: [{ id: 'Loras/other.gguf', scale: 1 }], reload: false })).status).toBe(200)
      expect(saved[llama.id].adapters).toEqual([{ id: 'Loras/other.gguf', scale: 1 }])
      expect(loads).toHaveLength(1)

      expect((await post({ config: { adapters: [] } })).status).toBe(200)
      expect(saved[llama.id].adapters).toEqual([])
      expect(loads).toHaveLength(2)
    } finally {
      await server.stop()
    }
  })

  it('helpers: choices need a known architecture; the patch schema takes only {id, scale}', () => {
    expect(adapterChoiceList({ ...llama, info: null }, adapters)).toEqual([])
    expect(parseRemoteLoadPatch({ adapters: [{ id: 'x', scale: 1, extra: 1 }] }).ok).toBe(false)
    const r = adaptersFromRemote([{ id: 'Loras/style.gguf', scale: -9 }], llama, adapters)
    expect(r).toEqual({ ok: true, adapters: [{ id: 'Loras/style.gguf', scale: -4 }] })
  })
})

class FakeServer extends EventEmitter {
  stdout = new EventEmitter()
  stderr = new EventEmitter()
  pid = undefined
}

describe('InstanceManager adapters', () => {
  afterEach(() => vi.unstubAllGlobals())

  const make = (adapterList: AdapterEntry[], spawned: string[][], over: Partial<InstanceManagerDeps> = {}): InstanceManager =>
    new InstanceManager({
      engine: { resolveExe: async () => 'C:\\llama\\llama-server.exe' } as unknown as EngineService,
      getModel: () => llama,
      draftFor: () => undefined,
      getDefaults: () => DEFAULT_LOAD_CONFIG,
      unloadPreviousOnLoad: () => true,
      estimate: () => null,
      getAdapter: (id) => adapterList.find((a) => a.id === id),
      pidFile: join(mkdtempSync(join(tmpdir(), 'rulz-inst-')), 'pids.json'),
      onChange: () => undefined,
      onError: () => undefined,
      spawnServer: (_exe, args) => {
        spawned.push(args)
        return new FakeServer() as unknown as ChildProcess
      },
      killProcess: (proc) => setTimeout(() => proc.emit('exit', 0), 5),
      ...over
    })

  it('resolves ids in order and leaves out missing adapters and comma paths with a note', () => {
    const list = [adapterEntry('a.gguf'), adapterEntry('b,b.gguf')]
    const { loaded, notes } = resolveAdapters(
      [{ id: 'a.gguf', scale: 0.5 }, { id: 'gone.gguf', scale: 1 }, { id: 'b,b.gguf', scale: 1 }],
      (id) => list.find((a) => a.id === id)
    )
    expect(loaded).toEqual([{ id: 'a.gguf', name: 'a', path: join(root, 'a.gguf'), scale: 0.5 }])
    expect(notes).toHaveLength(2)
    expect(notes[0]).toMatch(/gone\.gguf.*not found/)
    expect(notes[1]).toMatch(/comma/)
  })

  it('passes --lora, lists the loaded adapters, and writes skip notes to the instance log', async () => {
    const spawned: string[][] = []
    const mgr = make([adapterEntry('a.gguf')], spawned)
    const cfg = { ...DEFAULT_LOAD_CONFIG, adapters: [{ id: 'a.gguf', scale: 0.5 }, { id: 'gone.gguf', scale: 1 }] }
    mgr.load(llama.id, cfg).catch(() => undefined)
    await vi.waitFor(() => expect(mgr.list()).toHaveLength(1))
    const inst = mgr.list()[0]
    expect(spawned[0].filter((_, i) => spawned[0][i - 1] === '--lora')).toEqual([join(root, 'a.gguf')])
    expect(inst.adapters).toEqual([{ id: 'a.gguf', name: 'a', path: join(root, 'a.gguf'), scale: 0.5 }])
    expect(mgr.getLog(inst.instanceId).some((l) => /gone\.gguf.*not found/.test(l))).toBe(true)
    await mgr.unloadAll()
  })

  it('treats the same model with other adapters as a different load', async () => {
    const spawned: string[][] = []
    const mgr = make([adapterEntry('a.gguf')], spawned)
    const first = mgr.load(llama.id, { ...DEFAULT_LOAD_CONFIG, adapters: [{ id: 'a.gguf', scale: 1 }] })
    first.catch(() => undefined)
    await vi.waitFor(() => expect(spawned).toHaveLength(1))
    // Same adapters: reused. Another strength: restarted.
    mgr.load(llama.id, { ...DEFAULT_LOAD_CONFIG, adapters: [{ id: 'a.gguf', scale: 1 }] }).catch(() => undefined)
    await new Promise((r) => setTimeout(r, 30))
    expect(spawned).toHaveLength(1)
    const second = mgr.load(llama.id, { ...DEFAULT_LOAD_CONFIG, adapters: [{ id: 'a.gguf', scale: 2 }] })
    second.catch(() => undefined)
    await vi.waitFor(() => expect(spawned).toHaveLength(2))
    await mgr.unloadAll()
  })

  const stubServer = (scalesPosted: unknown[], fail = false): void => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string, init?: { method?: string; body?: string }) => {
        if (url.endsWith('/lora-adapters')) {
          scalesPosted.push(JSON.parse(init!.body!))
          return fail ? new Response('no', { status: 500 }) : new Response('{}')
        }
        if (url.endsWith('/props')) return new Response(JSON.stringify({ n_ctx: 4096 }))
        return new Response('ok')
      })
    )
  }

  it('posts the default scales after the server is up and before it is ready, when one differs from 1', async () => {
    const posted: unknown[] = []
    stubServer(posted)
    const mgr = make([adapterEntry('a.gguf'), adapterEntry('b.gguf')], [])
    const inst = await mgr.load(llama.id, { ...DEFAULT_LOAD_CONFIG, adapters: [{ id: 'a.gguf', scale: 0.5 }, { id: 'b.gguf', scale: 1 }] })
    expect(posted).toEqual([[{ id: 0, scale: 0.5 }, { id: 1, scale: 1 }]])
    expect(inst.status).toBe('ready')
    await mgr.unloadAll()
  })

  it('posts nothing when every default scale is 1, or there are no adapters', async () => {
    const posted: unknown[] = []
    stubServer(posted)
    const mgr = make([adapterEntry('a.gguf')], [])
    await mgr.load(llama.id, { ...DEFAULT_LOAD_CONFIG, adapters: [{ id: 'a.gguf', scale: 1 }] })
    expect(posted).toEqual([])
    await mgr.unloadAll()
    await mgr.load(llama.id, DEFAULT_LOAD_CONFIG)
    expect(posted).toEqual([])
    await mgr.unloadAll()
  })

  it('logs a failed scale POST and still reports the instance ready', async () => {
    const posted: unknown[] = []
    stubServer(posted, true)
    const mgr = make([adapterEntry('a.gguf')], [])
    const inst = await mgr.load(llama.id, { ...DEFAULT_LOAD_CONFIG, adapters: [{ id: 'a.gguf', scale: 0.5 }] })
    expect(inst.status).toBe('ready')
    expect(mgr.getLog(inst.instanceId).some((l) => /adapter scales not set/.test(l))).toBe(true)
    await mgr.unloadAll()
  })
})

describe('converter helpers', () => {
  const tmp = (): string => mkdtempSync(join(tmpdir(), 'rulz-conv-'))
  const req = (over: Partial<AdapterConvertRequest> = {}): AdapterConvertRequest => ({
    peftDir: 'C:\\peft',
    baseDir: '',
    baseModelId: '',
    outType: 'f16',
    outName: 'x',
    ...over
  })

  it('builds the arguments: --base wins over --base-model-id', () => {
    expect(buildConvertArgs('S.py', req(), 'O.gguf')).toEqual(['S.py', 'C:\\peft', '--outfile', 'O.gguf', '--outtype', 'f16'])
    expect(buildConvertArgs('S.py', req({ baseModelId: ' org/model ' }), 'O.gguf').slice(-2)).toEqual(['--base-model-id', 'org/model'])
    expect(buildConvertArgs('S.py', req({ baseDir: 'C:\\base', baseModelId: 'org/model' }), 'O.gguf').slice(-2)).toEqual(['--base', 'C:\\base'])
    expect(buildConvertArgs('S.py', req({ baseDir: 'C:\\base', baseModelId: 'org/model' }), 'O.gguf')).not.toContain('--base-model-id')
  })

  it('sanitizes the output name to a plain .gguf file name', () => {
    expect(sanitizeOutName('style')).toBe('style.gguf')
    expect(sanitizeOutName('Style.GGUF')).toBe('Style.gguf')
    expect(sanitizeOutName('..\\..\\evil/name')).toBe('name.gguf')
    expect(sanitizeOutName('a<b>c:d"e|f?g*h')).toBe('abcdefgh.gguf')
    expect(sanitizeOutName('a..b')).toBe('a.b.gguf')
    expect(sanitizeOutName('  trailing. . ')).toBe('trailing.gguf')
    expect(sanitizeOutName('{ftype}')).toBe('ftype.gguf')
    for (const bad of ['', '   ', '..', '.gguf', 'CON', 'nul.gguf', 'a/', '<>']) expect(sanitizeOutName(bad)).toBe('')
  })

  it('finds the script by walking up from the exe, at most 6 levels', () => {
    const base = tmp()
    const deep = join(base, 'build-cuda', 'bin', 'Release')
    mkdirSync(deep, { recursive: true })
    const exe = join(deep, 'llama-server.exe')
    expect(findConverterScript(exe)).toBeNull()
    writeFileSync(join(base, 'convert_lora_to_gguf.py'), '')
    expect(findConverterScript(exe)).toBe(join(base, 'convert_lora_to_gguf.py'))
    // Seven folders below it is out of reach.
    const far = join(base, 'a', 'b', 'c', 'd', 'e', 'f', 'g')
    mkdirSync(far, { recursive: true })
    expect(findConverterScript(join(far, 'llama-server.exe'))).toBeNull()
    const six = join(base, 'a', 'b', 'c', 'd', 'e', 'f')
    expect(findConverterScript(join(six, 'llama-server.exe'))).toBe(join(base, 'convert_lora_to_gguf.py'))
  })

  describe('AdapterConverter', () => {
    const setup = (opts: { python?: boolean; script?: boolean } = {}) => {
      const dir = tmp()
      const modelsDir = join(dir, 'models')
      const peft = join(dir, 'peft')
      mkdirSync(modelsDir)
      mkdirSync(peft)
      writeFileSync(join(peft, 'adapter_config.json'), '{}')
      const logs: string[] = []
      const calls: { command: string; args: string[]; cwd: string }[] = []
      let converted = 0
      let behave: (proc: FakeServer, out: string) => void = () => undefined
      const conv = new AdapterConverter({
        resolveExe: async () => join(dir, 'llama-server.exe'),
        getModelsDir: () => modelsDir,
        findPython: async () => (opts.python === false ? null : { command: 'python', args: [], version: '3.12.0' }),
        findScript: () => (opts.script === false ? null : join(dir, 'convert_lora_to_gguf.py')),
        onLog: (l) => logs.push(l),
        onConverted: () => void converted++,
        spawnProcess: (command, args, cwd) => {
          calls.push({ command, args, cwd })
          const proc = new FakeServer()
          const out = args[args.indexOf('--outfile') + 1]
          setTimeout(() => behave(proc, out), 5)
          return proc as unknown as ChildProcess
        }
      })
      return { dir, modelsDir, peft, logs, calls, conv, converted: () => converted, behave: (f: typeof behave) => (behave = f) }
    }
    const request = (peft: string, over: Partial<AdapterConvertRequest> = {}): AdapterConvertRequest => ({ peftDir: peft, baseDir: '', baseModelId: 'org/m', outType: 'f16', outName: 'style', ...over })

    it('reports what is missing in plain words', async () => {
      expect((await setup({ python: false }).conv.status()).error).toMatch(/Python/)
      expect((await setup({ script: false }).conv.status()).error).toMatch(/convert_lora_to_gguf\.py was not found/)
      const ok = await setup().conv.status()
      expect(ok).toMatchObject({ available: true, python: 'python' })
      expect(ok.error).toBeUndefined()
    })

    it('runs the converter in the script folder, streams lines, rescans and returns the path', async () => {
      const s = setup()
      s.behave((proc, out) => {
        proc.stdout.emit('data', 'INFO:one\r\nINFO:tw')
        proc.stderr.emit('data', 'warn\n')
        proc.stdout.emit('data', 'o\n')
        writeFileSync(out, 'gguf')
        proc.emit('exit', 0)
      })
      const r = await s.conv.convert(request(s.peft))
      expect(r).toMatchObject({ ok: true, outPath: join(s.modelsDir, 'style.gguf') })
      expect(s.logs.slice(1)).toEqual(['INFO:one', 'warn', 'INFO:two'])
      expect(r.logTail).toContain('INFO:two')
      expect(s.converted()).toBe(1)
      expect(s.calls[0].cwd).toBe(s.dir)
      expect(s.calls[0].args).toEqual([join(s.dir, 'convert_lora_to_gguf.py'), s.peft, '--outfile', join(s.modelsDir, 'style.gguf'), '--outtype', 'f16', '--base-model-id', 'org/m'])
      expect(s.calls[0].args).not.toContain('-I')
    })

    it('keeps the last 40 lines', async () => {
      const s = setup()
      s.behave((proc, out) => {
        proc.stdout.emit('data', Array.from({ length: 100 }, (_, i) => `line ${i}`).join('\n') + '\n')
        writeFileSync(out, 'x')
        proc.emit('exit', 0)
      })
      const r = await s.conv.convert(request(s.peft))
      expect(r.logTail).toHaveLength(40)
      expect(r.logTail.at(-1)).toBe('line 99')
    })

    it('deletes the partial output and returns the log tail when the converter fails', async () => {
      const s = setup()
      s.behave((proc, out) => {
        writeFileSync(out, 'partial')
        proc.stderr.emit('data', 'ModuleNotFoundError: torch\n')
        proc.emit('exit', 1)
      })
      const r = await s.conv.convert(request(s.peft))
      expect(r.ok).toBe(false)
      expect(r.error).toMatch(/exit code 1/)
      expect(r.logTail).toContain('ModuleNotFoundError: torch')
      expect(existsSync(join(s.modelsDir, 'style.gguf'))).toBe(false)
      expect(s.converted()).toBe(0)
    })

    it('refuses an existing output, a bad adapter folder, a bad base folder and an empty name without starting Python', async () => {
      const s = setup()
      writeFileSync(join(s.modelsDir, 'style.gguf'), 'mine')
      expect((await s.conv.convert(request(s.peft))).error).toMatch(/already exists/)
      expect((await s.conv.convert(request(join(s.dir, 'nope')))).error).toMatch(/adapter_config\.json/)
      expect((await s.conv.convert(request(s.peft, { baseDir: s.dir, outName: 'other' }))).error).toMatch(/config\.json/)
      expect((await s.conv.convert(request(s.peft, { outName: '..' }))).error).toMatch(/file name/)
      expect(s.calls).toHaveLength(0)
      expect(existsSync(join(s.modelsDir, 'style.gguf'))).toBe(true)
    })

    it('runs one conversion at a time', async () => {
      const s = setup()
      s.behave((proc, out) => {
        writeFileSync(out, 'x')
        proc.emit('exit', 0)
      })
      const first = s.conv.convert(request(s.peft))
      const second = await s.conv.convert(request(s.peft, { outName: 'two' }))
      expect(second).toMatchObject({ ok: false })
      expect(second.error).toMatch(/still running/)
      expect((await first).ok).toBe(true)
    })
  })
})

describe('batch runs keep their adapters', () => {
  const run = {
    id: 'r1',
    items: [],
    status: 'done',
    adapters: [{ name: 'style', scale: 9 }, { name: '', scale: 1 }, { scale: 1 }, { name: 'ok', scale: 0.5 }]
  }
  it('cleans and keeps the recorded adapters', () => {
    expect(cleanRun(run)!.adapters).toEqual([
      { name: 'style', scale: 4 },
      { name: 'ok', scale: 0.5 }
    ])
    expect(cleanRun({ ...run, adapters: 'x' })).not.toHaveProperty('adapters')
    expect(cleanRun({ ...run, adapters: [] })).not.toHaveProperty('adapters')
  })
  it('saves and loads them through the store', async () => {
    const path = join(mkdtempSync(join(tmpdir(), 'rulz-batch-')), 'b.json')
    await new BatchStore(path).save(run)
    const again = await new BatchStore(path).list()
    expect(again[0].adapters).toEqual([
      { name: 'style', scale: 4 },
      { name: 'ok', scale: 0.5 }
    ])
  })
})
