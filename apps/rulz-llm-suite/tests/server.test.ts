import { describe, expect, it, vi } from 'vitest'
import type { ApiRequestLog, LoadConfig, LoadedInstance, MemoryEstimate, ModelEntry } from '@shared/types'
import type { InstanceManager } from '../src/main/engine/instances'
import { ApiServer } from '../src/main/server/apiServer'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import http from 'node:http'
import { parseLoadConfig } from '../src/main/settings/schema'
import { applyRemoteLoadPatch, authorized, overrideSampling, draftChoices, draftFileFromRemote, hoistSystemMessages, parseRemoteLoadPatch, parseUsage, pickModel, publicModelId, remoteLoadView, resolveModel, UsageReader } from '../src/main/server/helpers'

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

  it('reads a streamed Responses API response.completed event', () => {
    const sse =
      'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Hi"}\n\n' +
      'event: response.completed\ndata: {"type":"response.completed","response":{"id":"r","output":[],"usage":{"input_tokens":12,"output_tokens":34}}}\n\n'
    expect(parseUsage(sse, true)).toEqual({ promptTokens: 12, completionTokens: 34 })
  })

  it('reads Anthropic message_start and message_delta usage', () => {
    const sse =
      'event: message_start\ndata: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":50,"output_tokens":1}}}\n\n' +
      'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"x"}}\n\n' +
      'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":9}}\n\n'
    expect(parseUsage(sse, true)).toEqual({ promptTokens: 50, completionTokens: 9 })
  })
})

describe('UsageReader', () => {
  it('reads usage from a long stream split anywhere, including inside a UTF-8 character', () => {
    const deltas = Array.from({ length: 2000 }, () => 'data: {"choices":[{"delta":{"content":"héllo wörld ✓"}}]}\n\n').join('')
    const raw = Buffer.from(`${deltas}data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2000}}\n\ndata: [DONE]\n\n`, 'utf8')
    const reader = new UsageReader(true)
    for (let i = 0; i < raw.length; i += 7) reader.push(raw.subarray(i, i + 7))
    expect(reader.end()).toEqual({ promptTokens: 5, completionTokens: 2000 })
  })

  it('reads usage from a plain body far bigger than the old 16 KB tail, and keeps it for error messages', () => {
    const body = JSON.stringify({ choices: [{ message: { content: 'x'.repeat(200_000) } }], usage: { prompt_tokens: 3, completion_tokens: 4 } })
    const reader = new UsageReader(false)
    for (let i = 0; i < body.length; i += 4096) reader.push(Buffer.from(body.slice(i, i + 4096)))
    expect(reader.end()).toEqual({ promptTokens: 3, completionTokens: 4 })
    expect(reader.text()).toBe(body)
  })

  it('handles a final event without a trailing newline', () => {
    const reader = new UsageReader(true)
    reader.push('data: {"usage":{"prompt_tokens":1,"completion_tokens":2}}')
    expect(reader.end()).toEqual({ promptTokens: 1, completionTokens: 2 })
  })
})

describe('authorized', () => {
  it('accepts bearer or x-api-key, and anything when no key is set', () => {
    expect(authorized('', {})).toBe(true)
    expect(authorized('k', {})).toBe(false)
    expect(authorized('k', { authorization: 'Bearer k' })).toBe(true)
    expect(authorized('k', { 'x-api-key': 'k' })).toBe(true)
    expect(authorized('k', { authorization: 'Bearer nope' })).toBe(false)
    expect(authorized('k', { authorization: 'Bearer kk', 'x-api-key': '' })).toBe(false)
    expect(authorized('key', { authorization: 'Bearer nope', 'x-api-key': 'key' })).toBe(true)
  })
})

describe('tensorSplit check', () => {
  it('rejects long near-misses quickly instead of backtracking', () => {
    const attacks = [' '.repeat(100_000), `1${' '.repeat(100_000)}x`, `1,${' '.repeat(50_000)},`, '1'.repeat(100_000) + 'x', ' 1 ,'.repeat(5000)]
    for (const tensorSplit of attacks) {
      const t = performance.now()
      expect(parseRemoteLoadPatch({ tensorSplit }).ok).toBe(false)
      expect(performance.now() - t).toBeLessThan(50)
    }
  })

  it('still accepts valid splits and refuses junk', () => {
    for (const ok of ['', '  ', '3,1', ' 60 , 40 ', '1.5,2.25,0', Array(16).fill('1').join(',')]) {
      expect(parseRemoteLoadPatch({ tensorSplit: ok }).ok, ok).toBe(true)
    }
    for (const bad of ['3,,1', '3,1,', ',3', '1.', '.5', '-1', '3 1', '3,1 --log-file x', Array(17).fill('1').join(',')]) {
      expect(parseRemoteLoadPatch({ tensorSplit: bad }).ok, bad).toBe(false)
    }
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
    const instances = { list: () => [instance()], readyFor: () => instance(), acquire: () => () => undefined } as unknown as InstanceManager
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

describe('API server requests and model lifetimes', () => {
  const settings = (port: number, idleUnloadMinutes = 0) => ({
    autoStart: false,
    port,
    bind: 'localhost' as const,
    apiKey: '',
    cors: false,
    idleUnloadMinutes,
    defaultModelId: '',
    unknownModelFallback: true
  })
  const makeServer = (instances: unknown, port: number, log: ApiRequestLog[] = [], idleUnloadMinutes = 0): ApiServer =>
    new ApiServer({
      instances: instances as InstanceManager,
      getModels: () => models,
      getSettings: () => settings(port, idleUnloadMinutes),
      getLoadDefaults: () => ({ config: DEFAULT_LOAD_CONFIG, custom: false }),
      setLoadDefaults: async () => undefined,
      estimate: () => null,
      gpus: () => [],
      onStatus: () => undefined,
      onRequest: (e) => log.push(e)
    })
  const listen = async (handler: http.RequestListener): Promise<{ server: http.Server; port: number }> => {
    const server = http.createServer(handler)
    await new Promise<void>((r) => server.listen(0, '127.0.0.1', () => r()))
    return { server, port: (server.address() as { port: number }).port }
  }
  const randomPort = (): number => 43000 + Math.floor(Math.random() * 2000)
  const post = (port: number, body: unknown): http.ClientRequest => {
    const req = http.request({ host: '127.0.0.1', port, method: 'POST', path: '/v1/chat/completions', headers: { 'Content-Type': 'application/json' } })
    req.on('error', () => undefined)
    req.end(JSON.stringify(body))
    return req
  }

  it('does not send the request on when the client left while the model was loading', async () => {
    let upstreamRequests = 0
    const upstream = await listen((_req, res) => {
      upstreamRequests++
      res.end('{}')
    })
    let finishLoad: () => void = () => undefined
    const instance = { instanceId: 'i1', modelId: models[0].id, status: 'ready', port: upstream.port, config: DEFAULT_LOAD_CONFIG } as unknown as LoadedInstance
    const instances = {
      list: () => [],
      readyFor: () => undefined,
      ensure: () => new Promise((r) => (finishLoad = () => r({ instance, started: true }))),
      acquire: () => () => undefined
    }
    const log: ApiRequestLog[] = []
    const port = randomPort()
    const server = makeServer(instances, port, log)
    await server.start()
    try {
      const req = post(port, { model: 'gemma', messages: [] })
      await vi.waitFor(() => expect(log.at(-1)?.phase).toBe('loading'))
      req.destroy()
      await vi.waitFor(() => expect(log.at(-1)).toMatchObject({ phase: 'done', status: 499 }))
      finishLoad()
      await new Promise((r) => setTimeout(r, 50))
      expect(upstreamRequests).toBe(0)
      expect(log.at(-1)?.status).toBe(499)
      // The model it loaded is still the server's to unload when idle.
      expect(server.getStatus().jitInstances.map((j) => j.instanceId)).toEqual(['i1'])
    } finally {
      await server.stop()
      await new Promise<void>((r) => upstream.server.close(() => r()))
    }
  })

  it('logs a client that leaves mid-stream as 499, not as the upstream error that follows, and holds the model meanwhile', async () => {
    let upstreamClosed = false
    const upstream = await listen((req, res) => {
      res.writeHead(200, { 'Content-Type': 'text/event-stream' })
      res.write('data: {"choices":[{"delta":{"content":"a"}}]}\n\n')
      req.socket.on('close', () => (upstreamClosed = true))
    })
    const instance = { instanceId: 'i1', modelId: models[0].id, status: 'ready', port: upstream.port, config: DEFAULT_LOAD_CONFIG } as unknown as LoadedInstance
    let users = 0
    const instances = {
      list: () => [instance],
      readyFor: () => instance,
      acquire: () => {
        users++
        return () => users--
      }
    }
    const log: ApiRequestLog[] = []
    const port = randomPort()
    const server = makeServer(instances, port, log)
    await server.start()
    try {
      const req = post(port, { model: 'Qwen3.5-9B-Q8_0', stream: true, messages: [] })
      const res = await new Promise<http.IncomingMessage>((r) => req.on('response', r))
      await new Promise((r) => res.once('data', r))
      expect(users).toBe(1)
      req.destroy()
      await vi.waitFor(() => expect(upstreamClosed).toBe(true))
      await vi.waitFor(() => expect(users).toBe(0))
      expect(log.filter((e) => e.phase === 'done').map((e) => e.status)).toEqual([499])
    } finally {
      await server.stop()
      upstream.server.closeAllConnections()
      await new Promise<void>((r) => upstream.server.close(() => r()))
    }
  })

  it('uses a model the UI is loading as it is, and only unloads its own loads when idle', async () => {
    const started = new Map<string, boolean>([['ui', false], ['api', true]])
    const instances = {
      list: () => [{ instanceId: 'ui' }, { instanceId: 'api' }],
      ensure: async (modelId: string) => {
        const id = modelId === models[0].id ? 'ui' : 'api'
        return { instance: { instanceId: id, modelId }, started: started.get(id) }
      }
    }
    const server = makeServer(instances, randomPort())
    const ensureLoaded = (id: string): Promise<LoadedInstance> =>
      (server as unknown as { ensureLoaded: (id: string) => Promise<LoadedInstance> }).ensureLoaded(id)
    await ensureLoaded(models[0].id)
    await ensureLoaded(models[1].id)
    expect(server.getStatus().jitInstances.map((j) => j.instanceId)).toEqual(['api'])
  })

  it('skips idle-unloading an instance that is in use and measures idle time from its last use', async () => {
    const unloaded: string[] = []
    let idleSince: number | null = null
    const instances = {
      list: () => [{ instanceId: 'i1' }],
      idleSince: () => idleSince,
      unload: async (id: string) => void unloaded.push(id)
    }
    const server = makeServer(instances, randomPort(), [], 1)
    const internals = server as unknown as { jit: Map<string, number>; unloadIdle: () => Promise<void> }
    // Last used by the API long ago, but in use now (a long stream, or a chat in the app).
    internals.jit.set('i1', 0)
    await internals.unloadIdle()
    expect(unloaded).toEqual([])
    idleSince = Date.now() - 30_000
    await internals.unloadIdle()
    expect(unloaded).toEqual([])
    idleSince = Date.now() - 61_000
    await internals.unloadIdle()
    expect(unloaded).toEqual(['i1'])
    expect(server.getStatus().jitInstances).toEqual([])
  })
})
