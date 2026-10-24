import { describe, expect, it } from 'vitest'
import type { ModelEntry } from '@shared/types'
import { authorized, parseUsage, publicModelId, resolveModel } from '../src/main/server/helpers'

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
