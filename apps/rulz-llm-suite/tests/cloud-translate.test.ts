import { describe, expect, it } from 'vitest'
import type { CloudModelDef, CloudModelInfo, CloudOptions } from '@shared/cloud'
import fixture from './fixtures/openrouter-models.json'
import { ReasoningCache, callsKey, canonicalJson } from '../src/main/cloud/reasoningCache'
import { ReplyTranslator, cleanArguments, kindForStatus, mapReasoning, translateRequest, type TranslateFail, type TranslateOk } from '../src/main/cloud/translate'
import { pickInfo } from './helpers/cloudKit'

const OPTS: CloudOptions = { dailyLimitUsd: 1, allowTraining: false }

function defOf(slug: string, patch: Partial<CloudModelInfo> = {}, limits: { contextLimit?: number; replyLimit?: number } = {}): CloudModelDef {
  const info = pickInfo(fixture, slug, patch)
  return { slug, info, contextLimit: limits.contextLimit ?? 32768, replyLimit: limits.replyLimit ?? 8192, allowApps: false, addedAt: 0 }
}
const FABLE = (): CloudModelDef => defOf('anthropic/claude-fable-5.2')
const GEMINI = (): CloudModelDef => defOf('google/gemini-4-neon')
const LING = (): CloudModelDef => defOf('inclusionai/ling-3.2-flash:free')
const SCHEMA = (): CloudModelDef => defOf('inference-net/schematron-v2-turbo')

function run(body: unknown, def = FABLE(), opts = OPTS, cache = new ReasoningCache()): TranslateOk {
  const r = translateRequest(JSON.stringify(body), def, opts, cache)
  if (!r.ok) throw new Error(`refused: ${r.message}`)
  return r
}
function refused(body: unknown, def = FABLE()): TranslateFail {
  const r = translateRequest(typeof body === 'string' ? body : JSON.stringify(body), def, OPTS, new ReasoningCache())
  if (r.ok) throw new Error('should have been refused')
  return r
}
const U = [{ role: 'user', content: 'hi' }]

describe('request: whitelist and basics', () => {
  it('builds from scratch: only whitelisted fields pass', () => {
    const r = run({
      model: 'whatever',
      messages: U,
      stream: false,
      return_progress: true,
      lora: [{ id: 0, scale: 1 }],
      dry_multiplier: 0.8,
      xtc_probability: 0.5,
      logit_bias: [['a', -1]],
      grammar: 'root ::= x',
      cache_prompt: true,
      id_slot: 3,
      user: 'someone',
      plugins: [{ id: 'web' }],
      transforms: ['middle-out'],
      models: ['a/b'],
      route: 'fallback',
      reasoning_budget_message: 'hurry',
      cache_control: { type: 'ephemeral' }
    })
    expect(Object.keys(r.body).sort()).toEqual(['max_tokens', 'messages', 'model', 'provider', 'stream'].sort())
    expect(r.body.model).toBe('anthropic/claude-fable-5.2')
    expect(r.body.stream).toBe(true)
    expect(r.stream).toBe(false)
    expect(run({ messages: U, stream: true }).stream).toBe(true)
  })

  it('rejects unusable bodies', () => {
    expect(refused('nope').code).toBe('cloud_request')
    expect(refused('[]').status).toBe(400)
    expect(refused({ messages: [] }).message).toMatch(/messages/)
    expect(refused({ messages: [{ role: 'weird', content: 'x' }] }).message).toMatch(/weird/)
    expect(refused({ messages: ['x'] }).code).toBe('cloud_request')
    expect(refused({ messages: [{ role: 'tool', content: 'x' }] }).message).toMatch(/tool_call_id/)
  })
})

describe('request: samplers', () => {
  it('sends only what the model supports, and only non-neutral values', () => {
    const body = { messages: U, temperature: 0.7, top_p: 0.9, top_k: 40, min_p: 0.05, repeat_penalty: 1.1, presence_penalty: 0.2, frequency_penalty: 0.3, seed: 42, stop: ['END'] }
    const ling = run(body, LING()).body
    expect(ling).toMatchObject({ temperature: 0.7, top_p: 0.9, top_k: 40, presence_penalty: 0.2, frequency_penalty: 0.3, seed: 42, stop: ['END'] })
    expect(ling).not.toHaveProperty('min_p')
    expect(ling).not.toHaveProperty('repetition_penalty')
    const schema = run(body, SCHEMA()).body
    expect(schema).toMatchObject({ temperature: 0.7, top_k: 40, min_p: 0.05, repetition_penalty: 1.1, seed: 42, stop: ['END'] })
    const fable = run(body, FABLE()).body
    expect(fable).not.toHaveProperty('temperature')
    expect(fable.stop).toEqual(['END'])
    expect(fable).not.toHaveProperty('seed')
  })

  it('omits neutral values', () => {
    const b = run({ messages: U, top_k: 0, min_p: 0, repeat_penalty: 1, presence_penalty: 0, frequency_penalty: 0, seed: null, stop: [] }, SCHEMA()).body
    for (const k of ['top_k', 'min_p', 'repetition_penalty', 'presence_penalty', 'frequency_penalty', 'seed', 'stop']) expect(b).not.toHaveProperty(k)
  })

  it('stop: strings only, at most four; a lone string is a list', () => {
    expect(run({ messages: U, stop: 'x' }, SCHEMA()).body.stop).toEqual(['x'])
    expect(run({ messages: U, stop: ['a', '', 3, 'b', 'c', 'd', 'e'] }, SCHEMA()).body.stop).toEqual(['a', 'b', 'c', 'd'])
  })

  it('accepts repetition_penalty as an API client sends it; ignores NaN-like and non-numeric values', () => {
    expect(run({ messages: U, repetition_penalty: 1.2 }, SCHEMA()).body.repetition_penalty).toBe(1.2)
    expect(run({ messages: U, temperature: 'hot' }, SCHEMA()).body).not.toHaveProperty('temperature')
  })
})

describe('request: max tokens', () => {
  it('min(requested or replyLimit, replyLimit, model max)', () => {
    expect(run({ messages: U }).maxTokens).toBe(8192)
    expect(run({ messages: U, max_tokens: 100 }).maxTokens).toBe(100)
    expect(run({ messages: U, max_tokens: 99999 }).maxTokens).toBe(8192)
    expect(run({ messages: U, max_completion_tokens: 50 }).maxTokens).toBe(50)
    expect(run({ messages: U, n_predict: 70 }).maxTokens).toBe(70)
    expect(run({ messages: U, n_predict: -1 }).maxTokens).toBe(8192)
    expect(run({ messages: U, max_tokens: 0 }).maxTokens).toBe(8192)
    expect(run({ messages: U, max_tokens: 12.9 }).maxTokens).toBe(12)
    expect(run({ messages: U }, defOf('anthropic/claude-fable-5.2', { maxReply: 1000 })).maxTokens).toBe(1000)
    expect(run({ messages: U }, FABLE() && defOf('anthropic/claude-fable-5.2', {}, { replyLimit: 4096 })).maxTokens).toBe(4096)
  })

  it('uses max_tokens, or max_completion_tokens when that is the only one supported', () => {
    expect(run({ messages: U, max_tokens: 10 }).body).toMatchObject({ max_tokens: 10 })
    const only = defOf('anthropic/claude-fable-5.2', { supported: ['max_completion_tokens', 'tools'] })
    const b = run({ messages: U, max_tokens: 10 }, only).body
    expect(b.max_completion_tokens).toBe(10)
    expect(b).not.toHaveProperty('max_tokens')
  })
})

describe('request: tools, images, response format', () => {
  const tool = { type: 'function', function: { name: 'get', description: 'd', parameters: { type: 'object', properties: {} }, strict: true }, cache_control: { type: 'ephemeral' } }
  it('passes tools, tool_choice and parallel_tool_calls in a clean form', () => {
    const b = run({ messages: U, tools: [tool, { type: 'web' }, null], tool_choice: 'auto', parallel_tool_calls: false }).body
    expect(b.tools).toEqual([{ type: 'function', function: { name: 'get', description: 'd', parameters: { type: 'object', properties: {} } } }])
    expect(b.tool_choice).toBe('auto')
    expect(b.parallel_tool_calls).toBe(false)
    expect(run({ messages: U, tool_choice: 'auto' }).body).not.toHaveProperty('tool_choice')
    expect(run({ messages: U, tools: [] }).body).not.toHaveProperty('tools')
  })
  it('tools to a model without tool support: 400 cloud_request with a sentence', () => {
    const r = refused({ messages: U, tools: [tool] }, SCHEMA())
    expect(r).toMatchObject({ status: 400, code: 'cloud_request' })
    expect(r.message).toMatch(/cannot use tools/)
  })
  it('images to a model without vision: 400 cloud_request; with vision they pass, cache_control stripped', () => {
    const img = { role: 'user', content: [{ type: 'text', text: 'what', cache_control: { type: 'ephemeral' } }, { type: 'image_url', image_url: { url: 'data:image/png;base64,AAAA', detail: 'low' } }] }
    expect(refused({ messages: [img] }, SCHEMA())).toMatchObject({ status: 400, code: 'cloud_request' })
    const r = run({ messages: [img] })
    expect(r.images).toBe(1)
    expect(r.body.messages).toEqual([{ role: 'user', content: [{ type: 'text', text: 'what' }, { type: 'image_url', image_url: { url: 'data:image/png;base64,AAAA', detail: 'low' } }] }])
    // The data URL is not counted as text.
    expect(r.inputTokens).toBeLessThan(20)
  })
  it('response_format: json_schema also requires parameters; json_object does not', () => {
    const js = { type: 'json_schema', json_schema: { name: 'out', schema: { type: 'object' }, strict: true, extra: 1 } }
    const a = run({ messages: U, response_format: js }).body
    expect(a.response_format).toEqual({ type: 'json_schema', json_schema: { name: 'out', schema: { type: 'object' }, strict: true } })
    expect((a.provider as any).require_parameters).toBe(true)
    const b = run({ messages: U, response_format: { type: 'json_object' } }).body
    expect(b.response_format).toEqual({ type: 'json_object' })
    expect(b.provider).not.toHaveProperty('require_parameters')
    expect(run({ messages: U, response_format: { type: 'text' } }).body).not.toHaveProperty('response_format')
    // A model that supports neither drops it.
    const none = defOf('inference-net/schematron-v2-turbo', { supported: ['max_tokens'] })
    expect(run({ messages: U, response_format: js }, none).body).not.toHaveProperty('response_format')
  })
})

describe('request: provider', () => {
  it('data_collection deny unless training is allowed; max_price at the dearest listed prices', () => {
    const def = defOf('openai/gpt-6.2-astra-pro')
    const p = run({ messages: U }, def).body.provider as any
    expect(p.data_collection).toBe('deny')
    expect(p.max_price.prompt).toBeGreaterThanOrEqual(20)
    expect(p.max_price.completion).toBeGreaterThanOrEqual(60)
    expect(p.max_price.prompt).toBeLessThan(20.001)
    expect(p.max_price).not.toHaveProperty('request')
    const q = run({ messages: U }, def, { dailyLimitUsd: 1, allowTraining: true }).body.provider as any
    expect(q).not.toHaveProperty('data_collection')
    expect(q.max_price).toBeDefined()
  })
  it('request and image prices go in USD when above 0', () => {
    const info = pickInfo(fixture, 'anthropic/claude-fable-5.2')
    const def = defOf('anthropic/claude-fable-5.2', { prices: { ...info.prices, request: 10000, image: 2000 } })
    const p = run({ messages: U }, def).body.provider as any
    expect(p.max_price.request).toBeCloseTo(0.01, 10)
    expect(p.max_price.image).toBeCloseTo(0.002, 10)
  })
})

describe('reasoning mapping', () => {
  const kw = (enable_thinking: boolean) => ({ chat_template_kwargs: { enable_thinking } })
  const withBudget = (supportsMaxTokens: boolean, mandatory = false) => defOf('inclusionai/ling-3.2-flash:free', { reasoning: { mandatory, efforts: ['low', 'high'], supportsMaxTokens } })

  it('nothing for auto or for a model without reasoning', () => {
    expect(run({ messages: U }).body).not.toHaveProperty('reasoning')
    expect(run({ messages: U, ...kw(false), reasoning_effort: 'high' }, SCHEMA()).body).not.toHaveProperty('reasoning')
  })
  it('off: {effort:none}, or nothing when mandatory', () => {
    expect(run({ messages: U, ...kw(false) }, LING()).body.reasoning).toEqual({ effort: 'none' })
    expect(run({ messages: U, ...kw(false) }, FABLE()).body).not.toHaveProperty('reasoning')
  })
  it('effort only when the model lists it', () => {
    expect(run({ messages: U, reasoning_effort: 'high' }).body.reasoning).toEqual({ effort: 'high' })
    expect(run({ messages: U, reasoning_effort: 'minimal' }).body).not.toHaveProperty('reasoning')
    expect(run({ messages: U, reasoning_effort: 'default' }).body).not.toHaveProperty('reasoning')
    // Not stated by the model: any known effort.
    expect(run({ messages: U, reasoning_effort: 'low' }, LING()).body.reasoning).toEqual({ effort: 'low' })
    expect(run({ messages: U, reasoning_effort: 'bogus' }, LING()).body).not.toHaveProperty('reasoning')
  })
  it('explicitly on sends enabled for a model that can be switched', () => {
    expect(run({ messages: U, ...kw(true) }, LING()).body.reasoning).toEqual({ enabled: true })
    expect(run({ messages: U, ...kw(true) }, FABLE()).body).not.toHaveProperty('reasoning')
  })
  it('budget: only when supported; below max_tokens and at least 1024, else left out', () => {
    expect(run({ messages: U, thinking_budget_tokens: 4000 }, withBudget(false)).body).not.toHaveProperty('reasoning')
    expect(run({ messages: U, thinking_budget_tokens: 4000, max_tokens: 10000 }, withBudget(true)).body.reasoning).toEqual({ max_tokens: 4000 })
    // Kept below max_tokens.
    expect(run({ messages: U, thinking_budget_tokens: 4000, max_tokens: 3000 }, withBudget(true)).body.reasoning).toEqual({ max_tokens: 2999 })
    // Too small to be worth it: left out (the effort, if any, is used instead).
    expect(run({ messages: U, thinking_budget_tokens: 4000, max_tokens: 1000 }, withBudget(true)).body).not.toHaveProperty('reasoning')
    expect(run({ messages: U, thinking_budget_tokens: 4000, max_tokens: 1000, reasoning_effort: 'low' }, withBudget(true)).body.reasoning).toEqual({ effort: 'low' })
    expect(run({ messages: U, thinking_budget_tokens: 500, max_tokens: 9000 }, withBudget(true)).body).not.toHaveProperty('reasoning')
    // Budget wins over effort: one or the other.
    expect(run({ messages: U, thinking_budget_tokens: 2048, reasoning_effort: 'high', max_tokens: 9000 }, withBudget(true)).body.reasoning).toEqual({ max_tokens: 2048 })
  })
  it('mapReasoning accepts a client reasoning object too', () => {
    expect(mapReasoning({ reasoning: { effort: 'medium' } }, FABLE(), 100)).toEqual({ effort: 'medium' })
    expect(mapReasoning({ reasoning: { effort: 'none' } }, LING(), 100)).toEqual({ effort: 'none' })
  })
})

describe('messages', () => {
  it('keeps role, content, tool_calls, tool_call_id, name, reasoning fields; drops the rest', () => {
    const r = run({
      messages: [
        { role: 'developer', content: 'be nice', cache_control: { type: 'ephemeral' }, junk: 1 },
        { role: 'user', content: 'hi', name: 'bob', foo: 'bar' },
        { role: 'assistant', content: null, reasoning_content: 'thinking', tool_calls: [{ id: 'c1', type: 'function', index: 0, function: { name: 'get', arguments: '{"a":1}' }, cache_control: {} }], timings: {} },
        { role: 'tool', tool_call_id: 'c1', name: 'get', content: 'ok', extra: 1 },
        { role: 'user', reasoning_content: 'ignored on user', content: 'x' }
      ]
    }).body.messages
    expect(r).toEqual([
      { role: 'system', content: 'be nice' },
      { role: 'user', content: 'hi', name: 'bob' },
      { role: 'assistant', content: null, tool_calls: [{ id: 'c1', type: 'function', function: { name: 'get', arguments: '{"a":1}' } }], reasoning_content: 'thinking' },
      { role: 'tool', tool_call_id: 'c1', content: 'ok', name: 'get' },
      { role: 'user', content: 'x' }
    ])
  })

  it('empty or invalid tool-call arguments become {}', () => {
    expect(cleanArguments('')).toBe('{}')
    expect(cleanArguments('   ')).toBe('{}')
    expect(cleanArguments('{"a":')).toBe('{}')
    expect(cleanArguments('[1,2]')).toBe('{}')
    expect(cleanArguments('42')).toBe('{}')
    expect(cleanArguments(undefined)).toBe('{}')
    expect(cleanArguments({ a: 1 })).toBe('{"a":1}')
    expect(cleanArguments('{"a": 1}')).toBe('{"a": 1}')
    const m = run({ messages: [{ role: 'assistant', content: '', tool_calls: [{ id: 'c', function: { name: 'f', arguments: '' } }, { function: { name: 'g', arguments: '{oops' } }, { function: {} }] }] }).body.messages as any[]
    expect(m[0].tool_calls).toEqual([
      { id: 'c', type: 'function', function: { name: 'f', arguments: '{}' } },
      { id: 'call_1', type: 'function', function: { name: 'g', arguments: '{}' } }
    ])
  })

  it('unknown content parts are dropped; text parts stay a list', () => {
    const m = run({ messages: [{ role: 'user', content: [{ type: 'input_audio', input_audio: {} }, { type: 'text', text: 'a' }] }] }).body.messages as any[]
    expect(m[0].content).toEqual([{ type: 'text', text: 'a' }])
  })
})

describe('reasoning cache across tool rounds', () => {
  const calls = [{ id: 'c1', name: 'get', arguments: '{"b":2,"a":1}' }]
  const details = [{ type: 'reasoning.encrypted', data: 'xyz', index: 0 }]

  it('canonical JSON ignores spacing and key order, and the key depends on slug, ids and names', () => {
    expect(canonicalJson('{"b": 2, "a": {"d":1,"c":2}}')).toBe('{"a":{"c":2,"d":1},"b":2}')
    expect(canonicalJson('not json')).toBe('not json')
    expect(callsKey('s', calls)).toBe(callsKey('s', [{ id: 'c1', name: 'get', arguments: '{ "a": 1, "b": 2 }' }]))
    expect(callsKey('s', calls)).not.toBe(callsKey('t', calls))
    expect(callsKey('s', calls)).not.toBe(callsKey('s', [{ id: 'c2', name: 'get', arguments: calls[0].arguments }]))
    expect(callsKey('s', calls)).not.toBe(callsKey('s', [{ id: 'c1', name: 'put', arguments: calls[0].arguments }]))
  })

  it('round trip: the reply\'s details come back on the assistant message, reasoning_content dropped', () => {
    const cache = new ReasoningCache()
    const tr = new ReplyTranslator('anthropic/claude-fable-5.2')
    tr.feed({ id: 'g', choices: [{ index: 0, delta: { reasoning_details: details, tool_calls: [{ index: 0, id: 'c1', function: { name: 'get', arguments: '{"b":2,' } }] } }] })
    tr.feed({ choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '"a":1}' } }] }, finish_reason: 'tool_calls' }] })
    tr.remember(cache)
    // The client sends the arguments re-serialised (other spacing, other key order) and its own reasoning_content.
    const r = run(
      {
        messages: [
          { role: 'user', content: 'go' },
          { role: 'assistant', content: '', reasoning_content: 'my thoughts', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'get', arguments: '{ "a": 1, "b": 2 }' } }] },
          { role: 'tool', tool_call_id: 'c1', content: 'r' }
        ]
      },
      FABLE(),
      OPTS,
      cache
    )
    const a = (r.body.messages as any[])[1]
    expect(a.reasoning_details).toEqual(details)
    expect(a).not.toHaveProperty('reasoning_content')
  })

  it('details a client sends itself win; another model gets none', () => {
    const cache = new ReasoningCache()
    cache.put('anthropic/claude-fable-5.2', calls, details)
    const mine = [{ type: 'reasoning.text', text: 'client', index: 0 }]
    const msg = (extra: object) => ({ messages: [{ role: 'assistant', content: '', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'get', arguments: calls[0].arguments } }], ...extra }] })
    expect(((run(msg({ reasoning_details: mine }), FABLE(), OPTS, cache).body.messages as any[])[0]).reasoning_details).toEqual(mine)
    const other = run(msg({ reasoning_content: 'rc' }), GEMINI(), OPTS, cache).body.messages as any[]
    expect(other[0]).not.toHaveProperty('reasoning_details')
    expect(other[0].reasoning_content).toBe('rc')
  })

  it('is bounded: least recently used goes first', () => {
    const cache = new ReasoningCache(1000)
    const big = (n: number) => [{ type: 'reasoning.text', text: 'x'.repeat(n) }]
    const k = (id: string) => [{ id, name: 'f', arguments: '{}' }]
    cache.put('s', k('a'), big(300))
    cache.put('s', k('b'), big(300))
    expect(cache.get('s', k('a'))).toBeDefined() // a is now the most recent
    cache.put('s', k('c'), big(300))
    expect(cache.get('s', k('b'))).toBeUndefined()
    expect(cache.get('s', k('a'))).toBeDefined()
    expect(cache.get('s', k('c'))).toBeDefined()
    cache.put('s', k('d'), big(5000)) // too big to keep
    expect(cache.get('s', k('d'))).toBeUndefined()
    expect(cache.size).toBeLessThanOrEqual(1000)
    cache.put('s', [], big(1))
    cache.put('s', k('e'), [])
    expect(cache.get('s', k('e'))).toBeUndefined()
  })
})

describe('reply translation', () => {
  const parse = (frames: string[]): any[] => frames.map((f) => JSON.parse(f.replace(/^data: /, '')))

  it('delta.reasoning becomes reasoning_content; details text is the fallback; details pass through', () => {
    const tr = new ReplyTranslator('s')
    const out = parse([
      ...tr.feed({ id: 'g1', model: 'm', created: 5, choices: [{ index: 0, delta: { role: 'assistant', content: '', reasoning: 'think ' } }] }),
      ...tr.feed({ choices: [{ index: 0, delta: { reasoning_details: [{ type: 'reasoning.text', text: 'more', index: 0 }] } }] }),
      ...tr.feed({ choices: [{ index: 0, delta: { reasoning: 'a', reasoning_details: [{ type: 'reasoning.text', text: 'dup', index: 1 }] } }] }),
      ...tr.feed({ choices: [{ index: 0, delta: { reasoning_details: [{ type: 'reasoning.summary', summary: 'sum', index: 2 }] } }] }),
      ...tr.feed({ choices: [{ index: 0, delta: { content: 'Hi' } }] })
    ])
    expect(out[0].choices[0].delta).toEqual({ role: 'assistant', reasoning_content: 'think ' })
    expect(out[1].choices[0].delta.reasoning_content).toBe('more')
    expect(out[1].choices[0].delta.reasoning_details).toEqual([{ type: 'reasoning.text', text: 'more', index: 0 }])
    expect(out[2].choices[0].delta.reasoning_content).toBe('a')
    expect(out[3].choices[0].delta.reasoning_content).toBe('sum')
    expect(out[4].choices[0].delta.content).toBe('Hi')
    expect(out[0]).toMatchObject({ id: 'g1', object: 'chat.completion.chunk', model: 'm', created: 5 })
    expect(tr.reasoning).toBe('think moreasum')
  })

  it('keeps tool_calls deltas as they are and collects them', () => {
    const tr = new ReplyTranslator('s')
    const d1 = [{ index: 0, id: 'c1', type: 'function', function: { name: 'get', arguments: '' } }]
    const d2 = [{ index: 0, function: { arguments: '{"a":1}' } }]
    const out = parse([...tr.feed({ choices: [{ index: 0, delta: { tool_calls: d1 } }] }), ...tr.feed({ choices: [{ index: 0, delta: { tool_calls: d2 }, finish_reason: 'tool_calls' }] })])
    expect(out[0].choices[0].delta.tool_calls).toEqual(d1)
    expect(out[1].choices[0].delta.tool_calls).toEqual(d2)
    expect(out[1].choices[0].finish_reason).toBe('tool_calls')
    expect(tr.callList()).toEqual([{ id: 'c1', name: 'get', arguments: '{"a":1}' }])
    expect(tr.collected()).toMatchObject({ object: 'chat.completion', choices: [{ message: { content: null, tool_calls: [{ id: 'c1', type: 'function', function: { name: 'get', arguments: '{"a":1}' } }] }, finish_reason: 'tool_calls' }] })
  })

  it('keeps usage with cost; BYOK adds the upstream cost; no cost means no usage', () => {
    const tr = new ReplyTranslator('s')
    const out = parse(tr.feed({ choices: [], usage: { prompt_tokens: 1, cost: 0.002, is_byok: true, cost_details: { upstream_inference_cost: 0.01 } } }))
    expect(out[0].choices).toEqual([])
    expect(out[0].usage.cost).toBeCloseTo(0.012, 10)
    expect(tr.costUsd).toBeCloseTo(0.012, 10)
    const plain = new ReplyTranslator('s')
    plain.feed({ choices: [], usage: { cost: 0.5, is_byok: false, cost_details: { upstream_inference_cost: 9 } } })
    expect(plain.costUsd).toBe(0.5)
    const none = new ReplyTranslator('s')
    none.feed({ choices: [], usage: { prompt_tokens: 3 } })
    expect(none.costUsd).toBeNull()
  })

  it('an error chunk ends the translation with a kind; empty chunks give nothing', () => {
    const tr = new ReplyTranslator('s')
    expect(tr.feed({ choices: [{ index: 0, delta: { content: '' } }] })).toEqual([])
    expect(tr.feed({ id: 'x', error: { code: 429, message: 'slow down' }, choices: [{ index: 0, delta: { content: '' }, finish_reason: 'error' }] })).toEqual([])
    expect(tr.error).toMatchObject({ code: 'cloud_rate', message: 'slow down' })
    const t2 = new ReplyTranslator('s')
    t2.feed({ choices: [{ index: 0, delta: {}, finish_reason: 'error' }] })
    expect(t2.error?.code).toBe('cloud_error')
    expect(kindForStatus(401, '')).toBe('cloud_key')
    expect(kindForStatus(402, '')).toBe('cloud_credits')
    expect(kindForStatus(403, 'Your input was flagged by moderation')).toBe('cloud_policy')
    expect(kindForStatus(403, 'Key limit exceeded')).toBe('cloud_credits')
    expect(kindForStatus(404, 'No endpoints found matching your data policy')).toBe('cloud_policy')
    expect(kindForStatus(503, 'No available provider under your price limits')).toBe('cloud_policy')
    expect(kindForStatus(503, 'overloaded')).toBe('cloud_error')
    expect(kindForStatus(400, '')).toBe('cloud_request')
    expect(kindForStatus(500, '')).toBe('cloud_error')
  })

  it('merges streamed details of one block when collected', () => {
    const tr = new ReplyTranslator('s')
    tr.feed({ choices: [{ index: 0, delta: { reasoning_details: [{ type: 'reasoning.text', text: 'a', index: 0 }] } }] })
    tr.feed({ choices: [{ index: 0, delta: { reasoning_details: [{ type: 'reasoning.text', text: 'b', index: 0, signature: 'sig' }] } }] })
    tr.feed({ choices: [{ index: 0, delta: { reasoning_details: [{ type: 'reasoning.encrypted', data: 'e', index: 1 }] } }] })
    expect(tr.details).toEqual([
      { type: 'reasoning.text', text: 'ab', index: 0, signature: 'sig' },
      { type: 'reasoning.encrypted', data: 'e', index: 1 }
    ])
    expect((tr.collected() as any).choices[0].message.reasoning_details).toHaveLength(2)
  })
})
