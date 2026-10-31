import { jsonifyToolCallArgs, templateStringifiesArgs } from '../src/main/chat/toolArgs'
import { describe, expect, it, vi } from 'vitest'
import { DEFAULT_SAMPLING } from '@shared/defaults'
import type { WireMessage } from '@shared/types'
import { ContextFullError, fitToContext, pinnedPrefixLength, truncateMiddle, upperBoundTokens } from '../src/main/chat/context'
import { buildRequestBody, DeltaBatcher } from '../src/main/chat/service'

describe('DeltaBatcher', () => {
  it('sends streamed text in batches and flushes on demand', () => {
    vi.useFakeTimers()
    try {
      const sent: [string, string][] = []
      const b = new DeltaBatcher((c, r) => sent.push([c, r]), 30)
      b.push('Hel', undefined)
      b.push('lo', 'hmm')
      expect(sent).toEqual([])
      vi.advanceTimersByTime(30)
      expect(sent).toEqual([['Hello', 'hmm']])
      b.push(' world', undefined)
      b.flush()
      b.flush()
      expect(sent).toEqual([['Hello', 'hmm'], [' world', '']])
      vi.advanceTimersByTime(100)
      expect(sent).toHaveLength(2)
    } finally {
      vi.useRealTimers()
    }
  })
})
import { errorFromBody, interpretChunk, SseParser, statsFromTimings } from '../src/main/chat/sse'
import { parseSettings } from '../src/main/settings/schema'
import { openAfterThinking } from '../src/renderer/src/lib/thinking'

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
  it('sends thinking settings only when they differ from the model default', () => {
    const plain = buildRequestBody('m', [], DEFAULT_SAMPLING)
    for (const key of ['chat_template_kwargs', 'reasoning_effort', 'thinking_budget_tokens', 'reasoning_budget_message']) {
      expect(plain).not.toHaveProperty(key)
    }
    const tuned = buildRequestBody('m', [], {
      ...DEFAULT_SAMPLING,
      thinking: 'on',
      reasoningEffort: 'high',
      thinkingBudget: 1024,
      thinkingBudgetMessage: 'Time to answer.'
    })
    expect(tuned).toMatchObject({
      chat_template_kwargs: { enable_thinking: true },
      reasoning_effort: 'high',
      thinking_budget_tokens: 1024,
      reasoning_budget_message: 'Time to answer.'
    })
  })
  it('turns thinking off, including when a handoff asks for it', () => {
    const off = buildRequestBody('m', [], { ...DEFAULT_SAMPLING, thinking: 'off', reasoningEffort: 'high', thinkingBudget: 10 })
    expect(off.chat_template_kwargs).toEqual({ enable_thinking: false })
    expect(off).not.toHaveProperty('reasoning_effort')
    expect(off).not.toHaveProperty('thinking_budget_tokens')
    expect(buildRequestBody('m', [], { ...DEFAULT_SAMPLING, thinking: 'on' }, true).chat_template_kwargs).toEqual({ enable_thinking: false })
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
    // Long enough that the quick size check can't rule out an overflow.
    const long = msgs.map((m) => ({ ...m, content: `${m.content}${' '.repeat(200)}` }))
    const fitted = await fitToContext(long, 500, 100, 'truncateMiddle', api)
    expect(fitted.map((m) => (m.content as string).trim())).toEqual(['sys', 'first', 'a2', 'last'])
    await expect(fitToContext(long, 500, 100, 'stop', api)).rejects.toBeInstanceOf(ContextFullError)
    expect(await fitToContext(long, 10000, 100, 'stop', api)).toBe(long)
  })

  it('skips counting when the chat is clearly small enough', async () => {
    let calls = 0
    const api = {
      countConversation: async () => {
        calls++
        return 10
      },
      countText: async () => 1
    }
    expect(await fitToContext(msgs, 8192, 2048, 'stop', api)).toBe(msgs)
    expect(calls).toBe(0)
    // Multi-byte text is counted by UTF-8 bytes, so it is never underestimated.
    expect(upperBoundTokens([{ role: 'user', content: '漢字' }])).toBe(64 + 16 + 6)
  })
})

describe('tool-call arguments for templates that print Python values', () => {
  const qwenCoder =
    "{{- '<parameter=' + args_name + '>\n' }}{%- set args_value = args_value if args_value is string else args_value | string %}{{- args_value }}"
  const fixed =
    "{{- '<parameter=' + args_name + '>\n' }}{%- set args_value = args_value | tojson | safe if args_value is mapping else args_value | string %}"

  it('spots only the XML-parameter templates that use | string', () => {
    expect(templateStringifiesArgs(qwenCoder)).toBe(true)
    expect(templateStringifiesArgs(fixed)).toBe(false)
    expect(templateStringifiesArgs('{{ tool_call.arguments | tojson }}')).toBe(false)
    expect(templateStringifiesArgs(undefined)).toBe(false)
  })

  it('turns non-string values into JSON text and leaves the rest alone', () => {
    const out = jsonifyToolCallArgs([
      { role: 'user', content: 'hi' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          { id: 'a', type: 'function', function: { name: 'load', arguments: '{"names":["x","y"],"deep":true,"path":"C:\\\\a"}' } },
          { id: 'b', type: 'function', function: { name: 'bad', arguments: 'not json' } }
        ]
      }
    ])
    const calls = out[1].tool_calls!
    expect(JSON.parse(calls[0].function.arguments)).toEqual({ names: '["x","y"]', deep: 'true', path: 'C:\\a' })
    expect(calls[1].function.arguments).toBe('not json')
    expect(out[0]).toEqual({ role: 'user', content: 'hi' })
  })
})

describe('reasoning block after thinking ends', () => {
  it('folds away by default', () => {
    expect(openAfterThinking(true, { defaultOpen: false, keepOpen: false })).toBe(false)
  })
  it('stays open with "Keep reasoning open after the answer" or "Expand reasoning blocks by default"', () => {
    expect(openAfterThinking(true, { defaultOpen: false, keepOpen: true })).toBe(true)
    expect(openAfterThinking(true, { defaultOpen: true, keepOpen: false })).toBe(true)
  })
  it('stays folded when the user closed it by hand', () => {
    expect(openAfterThinking(false, { defaultOpen: false, keepOpen: true })).toBe(false)
  })
  it('is off by default and saved with the other settings', () => {
    expect(parseSettings({}).keepThinkingOpen).toBe(false)
    expect(parseSettings({ keepThinkingOpen: true }).keepThinkingOpen).toBe(true)
    expect(parseSettings({ keepThinkingOpen: 'yes' }).keepThinkingOpen).toBe(false)
  })
})
