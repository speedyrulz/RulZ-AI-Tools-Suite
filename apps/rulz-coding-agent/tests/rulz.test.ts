import { splitThink } from '@shared/think'
import { afterAll, describe, expect, it } from 'vitest'
import { samplingRequest, skipsThinking } from '@shared/defaults'
import { findServer, parseLoadInfo, parseModels, pickDefaultModel, RulzClient } from '../src/main/rulz/client'
import { parseSettings } from '../src/main/settings'
import { errorFromBody, interpretChunk, SseParser, ToolCallAccumulator } from '../src/main/rulz/sse'
import { jsonifyToolCallArgs, templateStringifiesArgs } from '../src/main/rulz/toolArgs'
import { QWEN_CODER_TEMPLATE, startMockRulz } from './helpers/mockRulz'

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
  it('reads exact token counts from llama.cpp timings or an OpenAI usage block', () => {
    // prompt_n is only the part that wasn't already cached.
    expect(interpretChunk({ choices: [{ delta: {}, finish_reason: 'stop' }], timings: { cache_n: 1200, prompt_n: 34, predicted_n: 50 } }).usage).toEqual({
      prompt: 1234,
      completion: 50
    })
    expect(interpretChunk({ choices: [], usage: { prompt_tokens: 900, completion_tokens: 12 } }).usage).toEqual({ prompt: 900, completion: 12 })
    expect(interpretChunk({ choices: [{ delta: { content: 'hi' } }] }).usage).toBeUndefined()
  })
  it('reads thinking from every stream format', () => {
    // llama.cpp / DeepSeek
    expect(interpretChunk({ choices: [{ delta: { reasoning_content: 'plan' } }] })).toEqual({ reasoning: 'plan' })
    // vLLM, LM Studio, OpenRouter
    expect(interpretChunk({ choices: [{ delta: { reasoning: 'plan', content: null } }] })).toEqual({ reasoning: 'plan' })
    // Anthropic-style events
    expect(interpretChunk({ type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'plan' } })).toEqual({ reasoning: 'plan' })
    expect(interpretChunk({ type: 'content_block_delta', index: 1, delta: { type: 'text_delta', text: 'Done' } })).toEqual({ content: 'Done' })
    expect(interpretChunk({ type: 'content_block_delta', index: 0, delta: { type: 'signature_delta', signature: 'x' } })).toEqual({})
    // Inline <think> tags stay in the content; splitThink separates them, also mid-stream.
    const inline = ['<thi', 'nk>pl', 'an</think>Do', 'ne'].map((c) => interpretChunk({ choices: [{ delta: { content: c } }] }).content)
    expect(splitThink(inline.slice(0, 2).join(''))).toEqual({ reasoning: 'pl', answer: '' })
    expect(splitThink(inline.join(''))).toEqual({ reasoning: 'plan', answer: 'Done' })
  })
  it('reads error bodies', () => {
    expect(errorFromBody(400, '{"error":{"message":"context exceeded"}}')).toBe('context exceeded')
    expect(errorFromBody(500, '')).toBe('HTTP 500')
  })
})

describe('ToolCallAccumulator', () => {
  it('merges streamed fragments by index', () => {
    const acc = new ToolCallAccumulator()
    acc.push([{ index: 0, id: 'a', name: 'read_file' }])
    acc.push([{ index: 1, id: 'b', name: 'list_files', arguments: '{}' }])
    acc.push([{ index: 0, arguments: '{"path":' }])
    acc.push([{ index: 0, arguments: '"x.js"}' }])
    expect(acc.result()).toEqual([
      { id: 'a', type: 'function', function: { name: 'read_file', arguments: '{"path":"x.js"}' } },
      { id: 'b', type: 'function', function: { name: 'list_files', arguments: '{}' } }
    ])
  })
})

describe('Qwen3-Coder argument fix', () => {
  const fixed = "{{- '<parameter=' + args_name + '>\n' }}{%- set args_value = args_value | tojson | safe if args_value is mapping else args_value | string %}"

  it('spots only the XML-parameter templates that use | string', () => {
    expect(templateStringifiesArgs(QWEN_CODER_TEMPLATE)).toBe(true)
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

describe('telling RulZ apart from LM Studio', () => {
  it('recognises RulZ by its extra model fields', () => {
    expect(parseModels({ data: [{ id: 'a', owned_by: 'rulz', path: 'C:\\a.gguf', state: 'loaded', loaded_context_length: 4096 }] })).toEqual({
      isRulz: true,
      models: [{ id: 'a', loaded: true, loadedContext: 4096, maxContext: undefined, arch: undefined, quantization: undefined }]
    })
    expect(parseModels({ data: [{ id: 'a', owned_by: 'organization_owner' }] }).isRulz).toBe(false)
    expect(parseModels('nonsense').isRulz).toBe(false)
  })

  it('prefers a loaded Qwen3-Coder model', () => {
    const m = (id: string, loaded = false) => ({ id, loaded })
    expect(pickDefaultModel([m('gemma'), m('Qwen3-Coder-30B'), m('qwen3.6', true)])).toBe('qwen3.6')
    expect(pickDefaultModel([m('gemma', true), m('Qwen3-Coder-30B', true)])).toBe('Qwen3-Coder-30B')
    expect(pickDefaultModel([m('gemma'), m('Qwen3-Coder-30B')])).toBe('Qwen3-Coder-30B')
    expect(pickDefaultModel([])).toBe('')
  })

  const servers: { close: () => Promise<void> }[] = []
  afterAll(async () => Promise.all(servers.map((s) => s.close())))

  it('finds RulZ even when LM Studio answers first', async () => {
    const lm = await startMockRulz({ lmStudio: true })
    const rulz = await startMockRulz()
    servers.push(lm, rulz)
    const info = await findServer([lm.url, rulz.url], '')
    expect(info.status).toBe('connected')
    expect(info.url).toBe(rulz.url)
    expect(info.models[0].loaded).toBe(true)
  })

  it('explains when only another server answers, or nothing does', async () => {
    const lm = await startMockRulz({ lmStudio: true })
    servers.push(lm)
    expect(await findServer([lm.url], '')).toMatchObject({ status: 'other-server', otherUrl: lm.url })
    expect(await findServer(['http://127.0.0.1:9'], '')).toMatchObject({ status: 'not-found' })
  })

  it('asks for the API key when RulZ wants one', async () => {
    const locked = await startMockRulz({ apiKey: 'secret' })
    servers.push(locked)
    expect((await findServer([locked.url], '')).status).toBe('needs-key')
    expect((await findServer([locked.url], 'secret')).status).toBe('connected')
  })
})

describe('RulzClient', () => {
  it('streams text and tool calls, and applies the argument fix for Qwen3-Coder templates', async () => {
    const mock = await startMockRulz({
      turns: [{ content: 'Making it.', toolCalls: [{ name: 'write_file', args: { path: 'a.txt', content: 'hi' } }] }]
    })
    try {
      const client = new RulzClient(() => ({ url: mock.url, apiKey: '' }))
      const props = await client.modelProps('m')
      expect(props.nCtx).toBe(65536)
      const chunks: string[] = []
      const turn = await client.streamTurn(
        {
          model: 'm',
          messages: [
            { role: 'user', content: 'go' },
            { role: 'assistant', content: '', tool_calls: [{ id: 'x', type: 'function', function: { name: 'list_files', arguments: '{"depth":2}' } }] },
            { role: 'tool', tool_call_id: 'x', content: 'ok' }
          ]
        },
        new AbortController().signal,
        { onContent: (t) => chunks.push(t) }
      )
      expect(chunks.join('')).toBe('Making it.')
      expect(turn.toolCalls).toHaveLength(1)
      expect(JSON.parse(turn.toolCalls[0].function.arguments)).toEqual({ path: 'a.txt', content: 'hi' })
      expect(turn.finishReason).toBe('tool_calls')
      // The earlier call's number argument was sent as text.
      const sent = mock.requests[0].body.messages![1].tool_calls![0].function.arguments
      expect(JSON.parse(sent)).toEqual({ depth: '2' })
    } finally {
      await mock.close()
    }
  })
})

describe('RulzClient thinking', () => {
  it('streams reasoning_content to onReasoning before the answer', async () => {
    const mock = await startMockRulz({ turns: [{ reasoning: 'First I look at the files.', content: 'Done.' }] })
    try {
      const client = new RulzClient(() => ({ url: mock.url, apiKey: '' }))
      const events: string[] = []
      const turn = await client.streamTurn({ model: 'm', messages: [{ role: 'user', content: 'go' }] }, new AbortController().signal, {
        onReasoning: (t) => events.push(`r:${t}`),
        onContent: (t) => events.push(`c:${t}`)
      })
      const reasoning = events.filter((e) => e.startsWith('r:')).map((e) => e.slice(2))
      expect(reasoning.length).toBeGreaterThan(1)
      expect(reasoning.join('')).toBe('First I look at the files.')
      expect(events.findIndex((e) => e.startsWith('c:'))).toBeGreaterThan(events.findLastIndex((e) => e.startsWith('r:')))
      expect(turn.content).toBe('Done.')
    } finally {
      await mock.close()
    }
  })
})

describe('model settings', () => {
  it('lays the owner settings settings over the recommended ones', () => {
    expect(samplingRequest('Qwen3-Coder-30B', undefined)).toEqual({ temperature: 0.7, top_p: 0.8, top_k: 20, repeat_penalty: 1.05 })
    expect(samplingRequest('Qwen3-Coder-30B', { temperature: 0.2 })).toMatchObject({ temperature: 0.2, top_k: 20 })
    expect(samplingRequest('gemma-4', undefined)).toEqual({})
    expect(samplingRequest('gemma-4', { topP: 0.9, maxOutputTokens: 4096 })).toEqual({ top_p: 0.9 })
  })

  it('sends thinking settings like RulZ LLM Suite’s chat', () => {
    expect(samplingRequest('gemma-4', { thinking: 'on', reasoningEffort: 'high', thinkingBudget: 2048, thinkingBudgetMessage: 'Time to answer.' })).toEqual({
      chat_template_kwargs: { enable_thinking: true },
      reasoning_effort: 'high',
      thinking_budget_tokens: 2048,
      reasoning_budget_message: 'Time to answer.'
    })
    // Off wins over the effort and budget; no thinking setting sends nothing.
    expect(samplingRequest('gemma-4', { thinking: 'off', reasoningEffort: 'high', thinkingBudget: 10 })).toEqual({ chat_template_kwargs: { enable_thinking: false } })
    expect(samplingRequest('gemma-4', { reasoningEffort: 'low' })).toEqual({ reasoning_effort: 'low' })
    expect(skipsThinking({ thinkingBudget: 0 })).toBe(true)
    expect(skipsThinking({ thinking: 'off' })).toBe(true)
    expect(skipsThinking({ thinkingBudget: 2048 })).toBe(false)
    expect(skipsThinking(undefined)).toBe(false)
  })

  it('drops bad saved values one at a time', () => {
    const s = parseSettings({ modelSampling: { m: { temperature: 9, topK: 40, maxOutputTokens: 10 } } })
    expect(s.modelSampling).toEqual({ m: { topK: 40 } })
    expect(parseSettings({ modelSampling: 'nope' }).modelSampling).toEqual({})
    expect(parseSettings({ modelSampling: { m: { thinking: 'maybe', reasoningEffort: 'high' } } }).modelSampling).toEqual({ m: { reasoningEffort: 'high' } })
  })

  it('reads load settings and explains an older RulZ LLM Suite', async () => {
    expect(() => parseLoadInfo({})).toThrow()
    expect(parseLoadInfo({ config: { contextLength: 8192 }, custom: true, state: 'ready', loaded_context_length: 8192 })).toMatchObject({
      custom: true,
      state: 'ready',
      loadedContext: 8192,
      gpus: []
    })
    expect(parseLoadInfo({ config: {} }).draftChoices).toBeUndefined()
    const withDrafts = parseLoadInfo({
      config: { draftFile: 'builtin' },
      default_draft: { name: 'Built-in MTP layers', kind: 'mtp', built_in: true },
      draft_choices: [
        { id: 'builtin', name: 'Built-in MTP layers', folder: '', kind: 'mtp', built_in: true },
        { id: 'Gemma/Q5/mtp-gemma.gguf', name: 'mtp-gemma.gguf', folder: 'Gemma/Q5', kind: 'mtp', built_in: false },
        { name: 'no id' }
      ]
    })
    expect(withDrafts.defaultDraft).toEqual({ name: 'Built-in MTP layers', builtIn: true })
    expect(withDrafts.draftChoices?.map((c) => [c.id, c.builtIn])).toEqual([
      ['builtin', true],
      ['Gemma/Q5/mtp-gemma.gguf', false]
    ])
    expect(parseLoadInfo({ config: {}, default_draft: null, draft_choices: [] }).defaultDraft).toBeNull()
    const mock = await startMockRulz()
    try {
      const client = new RulzClient(() => ({ url: mock.url, apiKey: '' }))
      await expect(client.loadSettings('m')).rejects.toThrow(/Update RulZ LLM Suite/)
    } finally {
      await mock.close()
    }
  })
})

describe('splitThink', () => {
  it('separates inline thinking from the answer, even mid-stream', () => {
    expect(splitThink('<think>plan</think>\nAnswer')).toEqual({ reasoning: 'plan', answer: 'Answer' })
    expect(splitThink('<think>still going')).toEqual({ reasoning: 'still going', answer: '' })
    expect(splitThink('opening tag was in the prompt</think>Done')).toEqual({ reasoning: 'opening tag was in the prompt', answer: 'Done' })
    expect(splitThink('Plain reply')).toEqual({ reasoning: '', answer: 'Plain reply' })
  })
})
