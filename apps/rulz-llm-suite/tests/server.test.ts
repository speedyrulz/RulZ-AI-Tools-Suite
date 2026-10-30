import { describe, expect, it } from 'vitest'
import type { LoadConfig, LoadedInstance, MemoryEstimate, ModelEntry } from '@shared/types'
import type { InstanceManager } from '../src/main/engine/instances'
import { ApiServer } from '../src/main/server/apiServer'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import http from 'node:http'
import { parseLoadConfig } from '../src/main/settings/schema'
import { applyRemoteLoadPatch, authorized, overrideSampling, draftChoices, draftFileFromRemote, hoistSystemMessages, parseRemoteLoadPatch, parseUsage, pickModel, publicModelId, remoteLoadView, resolveModel } from '../src/main/server/helpers'

const model = (id: string, name: string): ModelEntry => ({
  id,
  name,
  folder: id.split('/').slice(0, -1).join('/'),
  path: `C:\\LLM\\${id.replace(/\//g, '\\')}`,
  files: [],
  sizeBytes: 1,
  info: null,
  modifiedAt: 0
})
const models = [
  model('Qwen3.5/Q8/Qwen3.5-9B-Q8_0.gguf', 'Qwen3.5-9B-Q8_0'),
  model('Gemma/gemma-4-31B-Q5_K_M.gguf', 'gemma-4-31B-Q5_K_M'),
  model('A/same.gguf', 'same'),
  model('B/same.gguf', 'same')
]

describe('publicModelId', () => {
  it('uses the name unless two models share it', () => {
    expect(publicModelId(models[0], models)).toBe('Qwen3.5-9B-Q8_0')
    expect(publicModelId(models[2], models)).toBe('A/same.gguf')
  })
})

describe('pickModel', () => {
  const on = { defaultModelId: '', unknownModelFallback: true }
  const off = { defaultModelId: '', unknownModelFallback: false }
  it('uses a model the library has, whatever is loaded', () => {
    expect(pickModel(models, 'gemma', [models[0].id], on)).toEqual({ model: models[1], fallback: false })
  })
  it('answers an unknown name with the single loaded model when the fallback is on', () => {
    expect(pickModel(models, 'claude-sonnet-5-5', [models[0].id], on)).toEqual({ model: models[0], fallback: true })
    expect(pickModel(models, 'claude-sonnet-5-5', [models[0].id], off)).toBeUndefined()
  })
  it('falls back to the default model when none or several are loaded', () => {
    const settings = { ...on, defaultModelId: models[1].id }
    expect(pickModel(models, 'claude-haiku-5-5', [], settings)).toEqual({ model: models[1], fallback: true })
    expect(pickModel(models, 'claude-haiku-5-5', [models[0].id, models[2].id], settings)?.model).toBe(models[1])
    expect(pickModel(models, 'claude-haiku-5-5', [models[0].id, models[2].id], on)).toBeUndefined()
  })
  it('treats a request with no model as before, not as a fallback', () => {
    expect(pickModel(models, undefined, [models[0].id], off)).toEqual({ model: models[0], fallback: false })
    expect(pickModel(models, undefined, [], off)).toBeUndefined()
  })
})

describe('resolveModel', () => {
  it('matches ids, names, file names and unique substrings, case-insensitively', () => {
    expect(resolveModel(models, 'Qwen3.5-9B-Q8_0')?.id).toBe(models[0].id)
    expect(resolveModel(models, 'qwen3.5-9b-q8_0.gguf')?.id).toBe(models[0].id)
    expect(resolveModel(models, 'Gemma/gemma-4-31B-Q5_K_M.gguf')?.id).toBe(models[1].id)
    expect(resolveModel(models, 'gemma')?.id).toBe(models[1].id)
    expect(resolveModel(models, 'A/same.gguf')?.id).toBe('A/same.gguf')
  })
  it('refuses ambiguous or missing names', () => {
    expect(resolveModel(models, 'same')).toBeUndefined()
    expect(resolveModel(models, 'q')).toBeUndefined()
    expect(resolveModel(models, '')).toBeUndefined()
    expect(resolveModel(models, 'nope')).toBeUndefined()
  })
})

describe('parseUsage', () => {
  it('reads OpenAI usage and llama.cpp timings from a JSON body', () => {
    expect(parseUsage('{"usage":{"prompt_tokens":10,"completion_tokens":5}}', false)).toEqual({ promptTokens: 10, completionTokens: 5 })
    expect(parseUsage('{"usage":{"input_tokens":3,"output_tokens":4},"timings":{"predicted_per_second":41.5}}', false)).toEqual({
      promptTokens: 3,
      completionTokens: 4,
      tokensPerSecond: 41.5
    })
  })
  it('reads the last data chunk of a stream, ignoring a cut-off first line', () => {
    const tail = 'oken":"x"}]}\ndata: {"choices":[],"timings":{"prompt_n":7,"cache_n":1,"predicted_n":20,"predicted_per_second":30}}\n\ndata: [DONE]\n'
    expect(parseUsage(tail, true)).toEqual({ promptTokens: 8, completionTokens: 20, tokensPerSecond: 30 })
  })
})

describe('authorized', () => {
  it('accepts bearer or x-api-key, and anything when no key is set', () => {
    expect(authorized('', {})).toBe(true)
    expect(authorized('k', {})).toBe(false)
    expect(authorized('k', { authorization: 'Bearer k' })).toBe(true)
    expect(authorized('k', { 'x-api-key': 'k' })).toBe(true)
    expect(authorized('k', { authorization: 'Bearer nope' })).toBe(false)
  })
})

describe('hoistSystemMessages', () => {
  it('moves system messages onto the end of the top-level system, in order', () => {
    const out = hoistSystemMessages({
      model: 'm',
      system: [{ type: 'text', text: 'core', cache_control: { type: 'ephemeral' } }],
      messages: [
        { role: 'user', content: 'hi' },
        { role: 'system', content: [{ type: 'text', text: '# Environment' }] },
        { role: 'assistant', content: 'hello' },
        { role: 'system', content: 'more' }
      ]
    } as never)
    expect(out).toEqual({
      model: 'm',
      system: [{ type: 'text', text: 'core', cache_control: { type: 'ephemeral' } }, { type: 'text', text: '# Environment' }, { type: 'text', text: 'more' }],
      messages: [
        { role: 'user', content: 'hi' },
        { role: 'assistant', content: 'hello' }
      ]
    })
  })
  it('accepts a string system or none, and leaves requests without system messages alone', () => {
    expect(hoistSystemMessages({ system: 'core', messages: [{ role: 'system', content: 'late' }, { role: 'user', content: 'hi' }] })).toEqual({
      system: [{ type: 'text', text: 'core' }, { type: 'text', text: 'late' }],
      messages: [{ role: 'user', content: 'hi' }]
    })
    expect(hoistSystemMessages({ messages: [{ role: 'user', content: 'hi' }, { role: 'system', content: 'late' }] })).toEqual({
      system: [{ type: 'text', text: 'late' }],
      messages: [{ role: 'user', content: 'hi' }]
    })
    expect(hoistSystemMessages({ system: 'core', messages: [{ role: 'user', content: 'hi' }] })).toBeNull()
    expect(hoistSystemMessages({ prompt: 'x' } as never)).toBeNull()
  })
})

describe('parseRemoteLoadPatch', () => {
  it('accepts the allowed load settings', () => {
    expect(parseRemoteLoadPatch({ contextLength: 32768, gpuLayers: 'all', cacheTypeK: 'q8_0' })).toEqual({
      ok: true,
      patch: { contextLength: 32768, gpuLayers: 'all', cacheTypeK: 'q8_0' }
    })
    expect(parseRemoteLoadPatch({ contextLength: null, gpuLayers: 40 })).toMatchObject({ ok: true })
  })

  it('refuses bad values and fields that are not exposed', () => {
    expect(parseRemoteLoadPatch({ contextLength: 10 })).toMatchObject({ ok: false })
    expect(parseRemoteLoadPatch({ cacheTypeV: 'q2' })).toMatchObject({ ok: false })
    expect(parseRemoteLoadPatch({ extraArgs: '--log-file x' })).toMatchObject({ ok: false })
    expect(parseRemoteLoadPatch({ chatTemplateFile: 'C:/x.jinja' })).toMatchObject({ ok: false })
    expect(parseRemoteLoadPatch({ tensorSplit: '3,1 --log-file x' })).toMatchObject({ ok: false })
    expect(parseRemoteLoadPatch({ draftPMin: 2 })).toMatchObject({ ok: false })
  })

  it('accepts GPU split and draft (MTP) settings', () => {
    expect(parseRemoteLoadPatch({ splitMode: 'custom', tensorSplit: '60, 40', draftTokens: 3, draftPMin: 0.75, draftGpu: 'cpu' })).toMatchObject({ ok: true })
    expect(parseRemoteLoadPatch({ tensorSplit: '' })).toMatchObject({ ok: true })
  })

  it('shows only the exposed fields', () => {
    const keys = Object.keys(remoteLoadView(DEFAULT_LOAD_CONFIG, models[0]))
    expect(keys).toContain('tensorSplit')
    expect(keys).toContain('draftCacheTypeV')
    expect(keys).not.toContain('extraArgs')
    expect(keys).not.toContain('chatTemplateFile')
  })

  it('offers draft choices by relative id and only accepts those', () => {
    const head = { path: 'C:\\LLM\\Gemma\\Q5\\mtp-gemma.gguf', sizeBytes: 2e8, kind: 'mtp' as const }
    const q5 = { ...model('Gemma/Q5/gemma-Q5.gguf', 'gemma-Q5'), draft: head, draftSidecar: head }
    const qwen = { ...model('Qwen/q-MTP.gguf', 'q-MTP'), info: { architecture: 'qwen35', hasChatTemplate: true, mtpLayers: 1 } }
    const library = [q5, qwen]
    expect(draftChoices(qwen, library).map((c) => c.id)).toEqual(['builtin', 'Gemma/Q5/mtp-gemma.gguf'])
    expect(draftChoices(q5, library)).toEqual([{ id: 'Gemma/Q5/mtp-gemma.gguf', name: 'mtp-gemma.gguf', folder: 'Gemma/Q5', kind: 'mtp', built_in: false }])
    expect(draftFileFromRemote('Gemma/Q5/mtp-gemma.gguf', qwen, library)).toBe(head.path)
    expect(draftFileFromRemote('builtin', qwen, library)).toBe('builtin')
    expect(draftFileFromRemote('builtin', q5, library)).toBeUndefined()
    expect(draftFileFromRemote('../../Windows/evil.gguf', qwen, library)).toBeUndefined()
    expect(draftFileFromRemote(null, qwen, library)).toBeNull()
    expect(remoteLoadView({ ...DEFAULT_LOAD_CONFIG, draftFile: head.path }, qwen).draftFile).toBe('Gemma/Q5/mtp-gemma.gguf')
  })
})

describe('GET/POST /v1/models/{id}/load-config', () => {
  it('reads and saves load defaults, and restarts a loaded model', async () => {
    const saved: Record<string, LoadConfig> = {}
    const loads: string[] = []
    const ready = { instanceId: 'i1', modelId: models[0].id, status: 'ready', config: DEFAULT_LOAD_CONFIG, contextLength: 8192 } as LoadedInstance
    const instances = {
      list: () => [ready],
      readyFor: () => ready,
      load: async (id: string) => {
        loads.push(id)
        return { ...ready, instanceId: 'i2', config: saved[id] }
      }
    } as unknown as InstanceManager
    const port = 41000 + Math.floor(Math.random() * 2000)
    const server = new ApiServer({
      instances,
      getModels: () => models,
      getSettings: () => ({ autoStart: false, port, bind: 'localhost', apiKey: '', cors: false, idleUnloadMinutes: 0, defaultModelId: '', unknownModelFallback: true }),
      getLoadDefaults: (id) => (saved[id] ? { config: saved[id], custom: true } : { config: DEFAULT_LOAD_CONFIG, custom: false }),
      setLoadDefaults: async (id, config) => void (saved[id] = config),
      estimate: (_m, config) => ({ gpuBytes: config.contextLength ?? 0 }) as unknown as MemoryEstimate,
      gpus: () => [{ index: 0, name: 'GPU A', memUsedMiB: 1, memTotalMiB: 24576, utilPct: 0 }],
      onStatus: () => undefined,
      onRequest: () => undefined
    })
    await server.start()
    try {
      const url = `http://127.0.0.1:${port}/v1/models/${encodeURIComponent('Qwen3.5-9B-Q8_0')}/load-config`
      const got = (await (await fetch(url)).json()) as { custom: boolean; state: string; config: { gpuLayers: unknown } }
      expect(got).toMatchObject({ custom: false, state: 'ready', config: { gpuLayers: DEFAULT_LOAD_CONFIG.gpuLayers } })

      const post = (body: unknown): Promise<Response> => fetch(url, { method: 'POST', body: JSON.stringify(body) })
      const res = await post({ config: { contextLength: 32768, gpuLayers: 'all' } })
      expect(res.status).toBe(200)
      expect(saved[models[0].id]).toMatchObject({ contextLength: 32768, gpuLayers: 'all', extraArgs: DEFAULT_LOAD_CONFIG.extraArgs })
      expect(loads).toEqual([models[0].id])

      const dry = (await (await post({ config: { contextLength: 65536 }, dry_run: true })).json()) as { estimate: { gpuBytes: number }; gpus: unknown[] }
      expect(dry.estimate.gpuBytes).toBe(65536)
      expect(dry.gpus).toHaveLength(1)
      expect(saved[models[0].id].contextLength).toBe(32768)
      expect(loads).toHaveLength(1)

      expect((await post({ config: { extraArgs: '--log-file C:/x' } })).status).toBe(400)
      expect((await post({ config: { contextLength: 4096 }, reload: false })).status).toBe(200)
      expect(loads).toHaveLength(1)
      expect((await fetch(url.replace('Qwen3.5-9B-Q8_0', 'nope'))).status).toBe(404)
    } finally {
      await server.stop()
    }
  })
})

describe('POST /v1/models/{id}/unload', () => {
  it('unloads every running instance of the model and only that model', async () => {
    let running = [
      { instanceId: 'i1', modelId: models[0].id, status: 'ready' },
      { instanceId: 'i2', modelId: models[0].id, status: 'loading' },
      { instanceId: 'i3', modelId: models[1].id, status: 'ready' }
    ] as LoadedInstance[]
    const unloaded: string[] = []
    const instances = {
      list: () => running,
      readyFor: (id: string) => running.find((i) => i.modelId === id && i.status === 'ready'),
      unload: async (id: string) => {
        unloaded.push(id)
        running = running.filter((i) => i.instanceId !== id)
      }
    } as unknown as InstanceManager
    const port = 41000 + Math.floor(Math.random() * 2000)
    const server = new ApiServer({
      instances,
      getModels: () => models,
      getSettings: () => ({ autoStart: false, port, bind: 'localhost', apiKey: '', cors: false, idleUnloadMinutes: 0, defaultModelId: '', unknownModelFallback: true }),
      getLoadDefaults: () => ({ config: DEFAULT_LOAD_CONFIG, custom: false }),
      setLoadDefaults: async () => undefined,
      estimate: () => null,
      gpus: () => [],
      onStatus: () => undefined,
      onRequest: () => undefined
    })
    await server.start()
    try {
      const unload = (id: string): Promise<Response> =>
        fetch(`http://127.0.0.1:${port}/v1/models/${encodeURIComponent(id)}/unload`, { method: 'POST' })
      const res = await unload('Qwen3.5-9B-Q8_0')
      expect(res.status).toBe(200)
      expect(await res.json()).toMatchObject({ state: 'not-loaded', unloaded: 2 })
      expect(unloaded.sort()).toEqual(['i1', 'i2'])
      expect(running.map((i) => i.instanceId)).toEqual(['i3'])

      expect(await (await unload('Qwen3.5-9B-Q8_0')).json()).toMatchObject({ unloaded: 0 })
      expect((await unload('nope')).status).toBe(404)
      expect((await fetch(`http://127.0.0.1:${port}/v1/models/x/unload`)).status).toBe(404)
    } finally {
      await server.stop()
    }
  })
})

describe('model sampling defaults', () => {
  const sampling = { ...DEFAULT_LOAD_CONFIG.sampling, temperature: 0.6, topK: 20, override: true }

  it('gives configs saved before sampling defaults existed empty ones', () => {
    const { sampling: _, ...old } = DEFAULT_LOAD_CONFIG
    expect(parseLoadConfig({ ...old, contextLength: 4096 }).sampling).toEqual(DEFAULT_LOAD_CONFIG.sampling)
    expect(parseLoadConfig({ ...old, sampling: { temperature: 99, topP: 0.9 } }).sampling).toMatchObject({ temperature: null, topP: 0.9, override: false })
  })

  it('accepts sampling changes over the API and merges them value by value', () => {
    const parsed = parseRemoteLoadPatch({ sampling: { topP: 0.9 } })
    expect(parsed.ok).toBe(true)
    if (!parsed.ok) return
    const config = { ...DEFAULT_LOAD_CONFIG, sampling }
    expect(applyRemoteLoadPatch(config, parsed.patch).sampling).toEqual({ ...sampling, topP: 0.9 })
    expect(parseRemoteLoadPatch({ sampling: { temperature: 9 } }).ok).toBe(false)
    expect(parseRemoteLoadPatch({ sampling: { seed: 1 } }).ok).toBe(false)
  })

  it('overrides only the values that are set', () => {
    expect(overrideSampling({ model: 'x', temperature: 1, top_p: 0.5 }, sampling)).toEqual({ model: 'x', temperature: 0.6, top_k: 20, top_p: 0.5 })
  })

  it('rewrites API requests for a model with override on, and leaves others alone', async () => {
    const seen: Record<string, unknown>[] = []
    const upstream = http.createServer((req, res) => {
      let raw = ''
      req.on('data', (c: Buffer) => (raw += c.toString('utf8')))
      req.on('end', () => {
        seen.push(JSON.parse(raw) as Record<string, unknown>)
        res.writeHead(200, { 'Content-Type': 'application/json' }).end('{"content":[],"usage":{"input_tokens":1,"output_tokens":1}}')
      })
    })
    await new Promise<void>((r) => upstream.listen(0, '127.0.0.1', () => r()))
    const upstreamPort = (upstream.address() as { port: number }).port
    let config = { ...DEFAULT_LOAD_CONFIG, sampling }
    const instance = (): LoadedInstance => ({ instanceId: 'i1', modelId: models[0].id, status: 'ready', port: upstreamPort, config, contextLength: 8192 }) as unknown as LoadedInstance
    const instances = { list: () => [instance()], readyFor: () => instance() } as unknown as InstanceManager
    const port = 41000 + Math.floor(Math.random() * 2000)
    const server = new ApiServer({
      instances,
      getModels: () => models,
      getSettings: () => ({ autoStart: false, port, bind: 'localhost', apiKey: '', cors: false, idleUnloadMinutes: 0, defaultModelId: '', unknownModelFallback: true }),
      getLoadDefaults: () => ({ config, custom: true }),
      setLoadDefaults: async () => undefined,
      estimate: () => null,
      gpus: () => [],
      onStatus: () => undefined,
      onRequest: () => undefined
    })
    await server.start()
    try {
      const send = (body: unknown): Promise<Response> =>
        fetch(`http://127.0.0.1:${port}/v1/messages`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      const request = { model: 'claude-sonnet-5-5', max_tokens: 8, temperature: 1, messages: [{ role: 'user', content: 'hi' }] }
      expect((await send(request)).status).toBe(200)
      expect(seen[0]).toMatchObject({ temperature: 0.6, top_k: 20, max_tokens: 8 })
      config = { ...config, sampling: { ...sampling, override: false } }
      await send(request)
      expect(seen[1]).toEqual(request)
    } finally {
      await server.stop()
      await new Promise<void>((r) => upstream.close(() => r()))
    }
  })
})
