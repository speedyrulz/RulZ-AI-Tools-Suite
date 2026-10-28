import { describe, expect, it } from 'vitest'
import type { LoadConfig, LoadedInstance, ModelEntry } from '@shared/types'
import type { InstanceManager } from '../src/main/engine/instances'
import { ApiServer } from '../src/main/server/apiServer'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import { authorized, parseRemoteLoadPatch, parseUsage, publicModelId, remoteLoadView, resolveModel } from '../src/main/server/helpers'

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
  })

  it('shows only the exposed fields', () => {
    expect(Object.keys(remoteLoadView(DEFAULT_LOAD_CONFIG)).sort()).toEqual(
      ['cacheTypeK', 'cacheTypeV', 'contextLength', 'flashAttention', 'gpuLayers', 'parallel', 'useDraft']
    )
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
      getSettings: () => ({ autoStart: false, port, bind: 'localhost', apiKey: '', cors: false, idleUnloadMinutes: 0, defaultModelId: '' }),
      getLoadDefaults: (id) => (saved[id] ? { config: saved[id], custom: true } : { config: DEFAULT_LOAD_CONFIG, custom: false }),
      setLoadDefaults: async (id, config) => void (saved[id] = config),
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

      expect((await post({ config: { extraArgs: '--log-file C:/x' } })).status).toBe(400)
      expect((await post({ config: { contextLength: 4096 }, reload: false })).status).toBe(200)
      expect(loads).toHaveLength(1)
      expect((await fetch(url.replace('Qwen3.5-9B-Q8_0', 'nope'))).status).toBe(404)
    } finally {
      await server.stop()
    }
  })
})
