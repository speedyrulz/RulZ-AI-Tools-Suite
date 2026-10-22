import { describe, expect, it } from 'vitest'
import { DEFAULT_SAMPLING } from '@shared/defaults'
import type { WireMessage } from '@shared/types'
import { ContextFullError, fitToContext, pinnedPrefixLength, truncateMiddle } from '../src/main/chat/context'
import { buildRequestBody } from '../src/main/chat/service'
import { errorFromBody, interpretChunk, SseParser, statsFromTimings } from '../src/main/chat/sse'

describe('SseParser', () => {
  it('handles payloads split across chunks', () => {
    const p = new SseParser()
    expect(p.push('data: {"a":1}\n\nda')).toEqual(['{"a":1}'])
    expect(p.push('ta: {"b":2}\r\n\ndata: [DONE]\n')).toEqual(['{"b":2}', '[DONE]'])
  })
  it('turns error: lines into error payloads', () => {
    const [payload] = new SseParser().push('error: {"message":"boom"}\n')
    expect(interpretChunk(JSON.parse(payload)).error).toBe('boom')
  })
})

describe('interpretChunk', () => {
  it('extracts content, reasoning, finish reason, timings and progress', () => {
    expect(interpretChunk({ choices: [{ delta: { reasoning_content: 'hmm' } }] })).toEqual({ reasoning: 'hmm' })
    expect(interpretChunk({ choices: [{ delta: { content: 'Hi' }, finish_reason: 'stop' }] })).toEqual({ content: 'Hi', finishReason: 'stop' })
    expect(interpretChunk({ choices: [], prompt_progress: { total: 110, cache: 10, processed: 60 } }).progress).toEqual({ processed: 50, total: 100 })
    const stats = statsFromTimings({ predicted_per_second: 42, predicted_n: 10, prompt_n: 5, cache_n: 3 }, {})
    expect(stats).toEqual({ tokensPerSecond: 42, predictedTokens: 10, promptTokens: 8, cachedTokens: 3 })
  })
  it('reads error bodies', () => {
    expect(errorFromBody(400, '{"error":{"message":"context exceeded"}}')).toBe('context exceeded')
    expect(errorFromBody(500, '')).toBe('HTTP 500')
  })
})

describe('buildRequestBody', () => {
  it('maps sampling params and optional fields', () => {
    const body = buildRequestBody('m', [], { ...DEFAULT_SAMPLING, maxTokens: 100, stop: ['###'], jsonSchema: '{"type":"object"}' })
    expect(body).toMatchObject({ model: 'm', stream: true, max_tokens: 100, stop: ['###'], top_k: 40, min_p: 0.05 })
    expect(body.response_format).toEqual({ type: 'json_schema', json_schema: { name: 'output', schema: { type: 'object' } } })
    expect(buildRequestBody('m', [], DEFAULT_SAMPLING)).not.toHaveProperty('max_tokens')
  })
})

describe('context fitting', () => {
  const msgs: WireMessage[] = [
    { role: 'system', content: 'sys' },
    { role: 'user', content: 'first' },
    { role: 'assistant', content: 'a1' },
    { role: 'user', content: 'u2' },
    { role: 'assistant', content: 'a2' },
    { role: 'user', content: 'last' }
  ]

  it('pins system prompt and first user message', () => {
    expect(pinnedPrefixLength(msgs)).toBe(2)
    expect(truncateMiddle([10, 10, 50, 50, 50, 10], 80, 2)).toEqual([0, 1, 4, 5])
  })

  it('drops middle messages until it fits, or throws for the stop policy', async () => {
    const api = {
      countConversation: async (m: WireMessage[]) => m.length * 100,
      countText: async () => 90
    }
    const fitted = await fitToContext(msgs, 500, 100, 'truncateMiddle', api)
    expect(fitted.map((m) => m.content)).toEqual(['sys', 'first', 'a2', 'last'])
    await expect(fitToContext(msgs, 500, 100, 'stop', api)).rejects.toBeInstanceOf(ContextFullError)
    expect(await fitToContext(msgs, 10000, 100, 'stop', api)).toBe(msgs)
  })
})
