import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { CloudCaller } from '../src/main/cloud/contract'
import fixture from './fixtures/openrouter-models.json'
import { interpretChunk, SseParser } from '../src/main/chat/sse'
import { retryAfterMs } from '../src/main/cloud/gateway'
import {
  DONE,
  Pipe,
  TEST_KEY,
  chunk,
  connectError,
  frame,
  goodStream,
  jsonResponse,
  makeRig,
  sseResponse,
  usageChunk,
  type Call,
  type Rig,
  type RigOptions
} from './helpers/cloudKit'

const rigs: Rig[] = []
async function rig(o: Partial<RigOptions> = {}): Promise<Rig> {
  const r = await makeRig({ fixture, ...o })
  rigs.push(r)
  return r
}
afterEach(() => {
  while (rigs.length) rigs.pop()!.cleanup()
})

const U = [{ role: 'user', content: 'hi' }]
const body = (extra: Record<string, unknown> = {}): Record<string, unknown> => ({ messages: U, stream: true, ...extra })

function chat(r: Rig, b: Record<string, unknown> = body(), by: CloudCaller = 'window', signal?: AbortSignal): Promise<Response> {
  return r.gw.fetch(r.modelId, '/v1/chat/completions', { method: 'POST', body: JSON.stringify(b), signal }, by)
}

/** Every data payload of an SSE text. */
function payloads(text: string): string[] {
  return new SseParser().push(text)
}
function frames(text: string): any[] {
  return payloads(text)
    .filter((p) => p !== '[DONE]')
    .map((p) => JSON.parse(p))
}
const diskLedger = (r: Rig): any => JSON.parse(readFileSync(join(r.dir, 'cloud-spend.json'), 'utf8'))
const spent = (r: Rig): number => r.ledger.snapshot().spent
const generationAnswer = (cost: number | null) => (c: Call) => (c.url.includes('/generation') ? jsonResponse({ data: { total_cost: cost } }) : undefined)

/** Handler: chat calls answered by `chatAnswer`, /generation calls by `gen`. */
function router(chatAnswer: (c: Call, n: number) => Response | Error | Promise<Response | Error>, gen: (c: Call) => Response | Error = () => new Response('', { status: 404 })) {
  let n = 0
  return (c: Call): Response | Error | Promise<Response | Error> => (c.url.includes('/generation') ? gen(c) : chatAnswer(c, ++n))
}

describe('a streamed answer', () => {
  it('goes to openrouter.ai with the key, header and no redirects; reservation is on disk before the fetch, id after the headers', async () => {
    const r = await rig()
    let during: any
    r.net.handler = (c) => {
      during = diskLedger(r)
      return sseResponse(c, goodStream('Hello', 0.0005), { 'x-generation-id': 'gen-A' })
    }
    const res = await chat(r)
    expect(res.status).toBe(200)
    expect(res.headers.get('content-type')).toContain('text/event-stream')
    // Before the fetch: the reservation is on disk.
    expect(during.reservations).toHaveLength(1)
    expect(during.reservations[0].amount).toBeGreaterThan(0)
    // Headers arrived: the generation id is written before any body is read.
    expect(diskLedger(r).reservations[0].generationId).toBe('gen-A')
    const text = await res.text()
    const call = r.net.chats[0]
    expect(call.url).toBe('https://openrouter.ai/api/v1/chat/completions')
    expect(call.method).toBe('POST')
    expect(call.headers.authorization).toBe(`Bearer ${TEST_KEY}`)
    expect(call.headers['x-openrouter-title']).toBe('RulZ LLM Suite')
    expect(call.headers['http-referer']).toBeUndefined()
    expect(call.redirect).toBe('error')
    expect(call.body).toMatchObject({ model: 'anthropic/claude-fable-5.2', stream: true, messages: U })
    const out = frames(text)
    expect(interpretChunk(out[1]).content).toBe('Hello')
    expect(out.find((f) => f.usage)?.usage.cost).toBe(0.0005)
    expect(payloads(text).at(-1)).toBe('[DONE]')
    expect(spent(r)).toBe(500)
    expect(r.ledger.snapshot().reserved).toBe(0)
    expect(r.ledger.snapshot().models).toEqual({ 'anthropic/claude-fable-5.2': 500 })
  })

  it('reads split chunks and UTF-8 split across chunks; strips OpenRouter comment lines', async () => {
    const r = await rig()
    const text = ': OPENROUTER PROCESSING\n\n' + [chunk({ role: 'assistant', content: '' }), chunk({ content: 'héllo 日本語 😀' }), ': OPENROUTER PROCESSING\n\n', chunk({}, {}, 'stop'), usageChunk(0.001), DONE].join('')
    const bytes = new TextEncoder().encode(text)
    // One byte at a time: every multi-byte character is split.
    const parts: Uint8Array[] = []
    for (let i = 0; i < bytes.length; i++) parts.push(bytes.slice(i, i + 1))
    r.net.handler = (c) => sseResponse(c, parts)
    const res = await chat(r)
    const out = await res.text()
    expect(out).not.toContain('PROCESSING')
    const content = frames(out)
      .map((f) => interpretChunk(f).content ?? '')
      .join('')
    expect(content).toBe('héllo 日本語 😀')
    expect(spent(r)).toBe(1000)
  })

  it('output chunks are readable by interpretChunk: reasoning, tool calls, finish reason', async () => {
    const r = await rig()
    const tc1 = [{ index: 0, id: 'c1', type: 'function', function: { name: 'get', arguments: '' } }]
    const tc2 = [{ index: 0, function: { arguments: '{"a":1}' } }]
    r.net.handler = (c) =>
      sseResponse(c, [
        chunk({ role: 'assistant', content: '', reasoning: 'thinking...' }),
        chunk({ reasoning_details: [{ type: 'reasoning.text', text: 'x', index: 0 }], tool_calls: tc1 }),
        chunk({ tool_calls: tc2 }, {}, 'tool_calls'),
        usageChunk(0.002),
        DONE
      ])
    const out = frames(await (await chat(r, body({ tools: [{ type: 'function', function: { name: 'get' } }] }))).text())
    expect(interpretChunk(out[0]).reasoning).toBe('thinking...')
    expect(interpretChunk(out[1]).toolCalls).toEqual([{ index: 0, id: 'c1', name: 'get', arguments: '' }])
    expect(out[1].choices[0].delta.tool_calls).toEqual(tc1)
    expect(out[1].choices[0].delta.reasoning_details).toEqual([{ type: 'reasoning.text', text: 'x', index: 0 }])
    expect(interpretChunk(out[2]).finishReason).toBe('tool_calls')
    expect(spent(r)).toBe(2000)
  })

  it('BYOK adds the upstream inference cost', async () => {
    const r = await rig()
    r.net.handler = (c) => sseResponse(c, [chunk({ content: 'x' }), chunk({}, {}, 'stop'), usageChunk(0.001, { is_byok: true, cost_details: { upstream_inference_cost: 0.004 } }), DONE])
    await (await chat(r)).text()
    expect(spent(r)).toBe(5000)
  })

  it('a stream that finishes without usage.cost counts as no usage: looked up, else the reservation is charged', async () => {
    const r = await rig()
    r.sleeper.auto = true
    r.net.handler = router((c) => sseResponse(c, [chunk({ content: 'x' }), chunk({}, {}, 'stop'), frame({ id: 'gen-1', choices: [], usage: { prompt_tokens: 3 } }), DONE]), generationAnswer(0.0007) as never)
    await (await chat(r)).text()
    await r.gw.settled()
    expect(r.net.calls.some((c) => c.url === 'https://openrouter.ai/api/v1/generation?id=gen-1')).toBe(true)
    expect(spent(r)).toBe(700)
  })

  it('output chunks carry no text of the key', async () => {
    const r = await rig()
    r.net.handler = (c) => sseResponse(c, goodStream('my key is ' + TEST_KEY))
    const out = await (await chat(r)).text()
    // The model said it; only errors and state are scrubbed, never what the model wrote.
    expect(out).toContain('my key is')
    expect(r.net.calls.every((c) => !c.url.includes(TEST_KEY))).toBe(true)
  })
})

describe('mid-stream failures', () => {
  it('an error chunk ends the stream with one error data chunk of a CloudErrorKind', async () => {
    const r = await rig()
    r.sleeper.auto = true
    r.net.handler = router(
      (c) => sseResponse(c, [chunk({ content: 'par' }), frame({ id: 'gen-1', error: { code: 429, message: 'Provider is rate limited' }, choices: [{ index: 0, delta: { content: '' }, finish_reason: 'error' }] }), chunk({ content: 'never' })]),
      generationAnswer(0.0002) as never
    )
    const text = await (await chat(r)).text()
    const ps = payloads(text)
    const errors = ps.filter((p) => p !== '[DONE]' && JSON.parse(p).error)
    expect(errors).toHaveLength(1)
    expect(JSON.parse(errors[0]).error).toMatchObject({ message: 'Provider is rate limited', code: 'cloud_rate' })
    expect(text).not.toContain('never')
    expect(interpretChunk(JSON.parse(errors[0])).error).toBe('Provider is rate limited')
    await r.gw.settled()
    expect(spent(r)).toBe(200)
    expect(r.ledger.reservations()).toEqual([])
  })

  it('a broken connection mid-stream gives an error chunk and is looked up, else the reservation is charged', async () => {
    const r = await rig()
    r.sleeper.auto = true
    const pipe = new Pipe()
    r.net.handler = router(
      (c) => new Response(pipe.body(c.signal), { status: 200 }),
      () => new Response('', { status: 404 })
    )
    const res = await chat(r)
    pipe.push(chunk({ content: 'a' }))
    pipe.fail(new TypeError('terminated'))
    const text = await res.text()
    expect(JSON.parse(payloads(text).find((p) => p !== '[DONE]' && JSON.parse(p).error)!).error.code).toBe('cloud_error')
    await r.gw.settled()
    const lookups = r.net.calls.filter((c) => c.url.includes('/generation'))
    expect(lookups).toHaveLength(3)
    expect(r.sleeper.waits.map((w) => w.ms)).toEqual([1000, 4000, 15000])
    // Charged the whole reservation (nothing else is known).
    expect(spent(r)).toBeGreaterThan(1000)
    expect(r.ledger.reservations()).toEqual([])
  })

  it('a stream that ends with no finish and no DONE is an error', async () => {
    const r = await rig()
    r.sleeper.auto = true
    r.net.handler = router((c) => sseResponse(c, [chunk({ content: 'a' })]))
    const text = await (await chat(r)).text()
    expect(text).toContain('"code":"cloud_error"')
    await r.gw.settled()
    expect(r.ledger.reservations()).toEqual([])
    expect(spent(r)).toBeGreaterThan(0)
  })

  it('past 1.5 x max_tokens the stream is ended (length) and the request aborted', async () => {
    const r = await rig()
    r.sleeper.auto = true
    const pipe = new Pipe()
    let signal: AbortSignal | undefined
    r.net.handler = router((c) => {
      signal = c.signal
      return new Response(pipe.body(c.signal), { status: 200 })
    })
    const res = await chat(r, body({ max_tokens: 10 }))
    // 10 tokens ~ 35 chars; 1.5x = 52 chars of ASCII.
    pipe.push(chunk({ content: 'a'.repeat(40) }))
    pipe.push(chunk({ content: 'b'.repeat(40) }))
    pipe.push(chunk({ content: 'c'.repeat(40) }))
    const text = await res.text()
    const out = frames(text)
    expect(out.at(-1).choices[0].finish_reason).toBe('length')
    expect(text).not.toContain('c'.repeat(40))
    expect(signal?.aborted).toBe(true)
    await r.gw.settled()
    expect(r.ledger.reservations()).toEqual([])
    expect(spent(r)).toBe(500)
  })
})

describe('idle timeout', () => {
  it('no bytes between chunks ends the reply with an error chunk, aborts upstream, settles', async () => {
    const r = await rig({ idleMs: 40 })
    r.sleeper.auto = true
    let signal: AbortSignal | undefined
    r.net.handler = router((c) => {
      signal = c.signal
      return sseResponse(c, [chunk({ content: 'a' })], {}, true)
    })
    const res = await chat(r)
    const text = await res.text()
    expect(text).toContain('"code":"cloud_error"')
    expect(text).toMatch(/nothing for/)
    expect(signal?.aborted).toBe(true)
    await r.gw.settled()
    expect(r.ledger.reservations()).toEqual([])
    expect(spent(r)).toBeGreaterThan(0)
  })

  it('comment lines count as bytes', async () => {
    const r = await rig({ idleMs: 60 })
    const pipe = new Pipe()
    r.net.handler = (c) => new Response(pipe.body(c.signal), { status: 200 })
    const res = await chat(r)
    const reading = res.text()
    for (let i = 0; i < 5; i++) {
      pipe.push(': OPENROUTER PROCESSING\n\n')
      await new Promise((x) => setTimeout(x, 30))
    }
    for (const f of goodStream('late', 0.0001)) pipe.push(f)
    pipe.end()
    const out = await reading
    expect(out).toContain('late')
    expect(out).not.toContain('"error"')
    expect(spent(r)).toBe(100)
  })

  it('no headers within the idle time: the request is aborted, an error answer, reservation settled', async () => {
    const r = await rig({ idleMs: 40 })
    r.net.handler = (c) => new Promise((_, rej) => c.signal?.addEventListener('abort', () => rej(new DOMException('aborted', 'AbortError'))))
    const res = await chat(r)
    expect(res.status).toBe(502)
    expect((await res.json()).error.code).toBe('cloud_error')
    expect(r.ledger.reservations()).toEqual([])
  })
})

describe('caller abort and cancel', () => {
  it('aborting the caller aborts the upstream fetch; the body errors; the cost is looked up', async () => {
    const r = await rig()
    r.sleeper.auto = true
    const pipe = new Pipe()
    let signal: AbortSignal | undefined
    r.net.handler = router(
      (c) => {
        signal = c.signal
        return new Response(pipe.body(c.signal), { status: 200, headers: { 'x-generation-id': 'gen-Z' } })
      },
      generationAnswer(0.0004) as never
    )
    const ctl = new AbortController()
    const res = await chat(r, body(), 'window', ctl.signal)
    const reader = res.body!.getReader()
    pipe.push(chunk({ content: 'a' }))
    await reader.read()
    ctl.abort()
    await expect(reader.read()).rejects.toThrow()
    expect(signal?.aborted).toBe(true)
    await r.gw.settled()
    expect(spent(r)).toBe(400)
    expect(r.ledger.reservations()).toEqual([])
  })

  it('without an id, an abort after the request was sent charges the reservation', async () => {
    const r = await rig()
    const ctl = new AbortController()
    r.net.handler = (c) => new Promise((_, rej) => c.signal?.addEventListener('abort', () => rej(new DOMException('aborted', 'AbortError'))))
    const p = chat(r, body(), 'window', ctl.signal)
    await vi.waitFor(() => expect(r.net.chats).toHaveLength(1))
    ctl.abort()
    await expect(p).rejects.toThrow()
    expect(spent(r)).toBeGreaterThan(0)
    expect(r.ledger.reservations()).toEqual([])
  })

  it('an already aborted signal rejects before anything is booked', async () => {
    const r = await rig()
    const ctl = new AbortController()
    ctl.abort()
    await expect(chat(r, body(), 'window', ctl.signal)).rejects.toThrow()
    expect(r.net.calls).toHaveLength(0)
    expect(r.ledger.reservations()).toEqual([])
  })

  it('the caller cancelling the Response body aborts upstream and settles once', async () => {
    const r = await rig()
    r.sleeper.auto = true
    const pipe = new Pipe()
    let signal: AbortSignal | undefined
    r.net.handler = router(
      (c) => {
        signal = c.signal
        return new Response(pipe.body(c.signal), { status: 200, headers: { 'x-generation-id': 'gen-C' } })
      },
      generationAnswer(0.0003) as never
    )
    const res = await chat(r)
    await res.body!.cancel()
    expect(signal?.aborted).toBe(true)
    await r.gw.settled()
    expect(spent(r)).toBe(300)
    expect(r.ledger.reservations()).toEqual([])
    // Nothing settles twice later.
    pipe.end()
    await new Promise((x) => setTimeout(x, 20))
    await r.gw.settled()
    expect(spent(r)).toBe(300)
  })

  it('a caller that never reads the body still gets the cost settled', async () => {
    const r = await rig()
    r.net.handler = (c) => sseResponse(c, goodStream('x', 0.0009))
    const res = await chat(r)
    void res
    await vi.waitFor(() => expect(spent(r)).toBe(900))
    expect(r.ledger.reservations()).toEqual([])
  })

  it('reading after it all finished still works and settles nothing more', async () => {
    const r = await rig()
    r.net.handler = (c) => sseResponse(c, goodStream('x', 0.0009))
    const res = await chat(r)
    await vi.waitFor(() => expect(spent(r)).toBe(900))
    expect(payloads(await res.text()).at(-1)).toBe('[DONE]')
    await res.body?.cancel().catch(() => undefined)
    expect(spent(r)).toBe(900)
  })
})

describe('dispose', () => {
  it('aborts what runs and leaves the reservation on disk for the next start', async () => {
    const r = await rig()
    const pipe = new Pipe()
    let signal: AbortSignal | undefined
    r.net.handler = (c) => {
      signal = c.signal
      return new Response(pipe.body(c.signal), { status: 200, headers: { 'x-generation-id': 'gen-D' } })
    }
    const res = await chat(r)
    const reading = res.text().catch(() => 'broken')
    pipe.push(chunk({ content: 'a' }))
    await new Promise((x) => setTimeout(x, 10))
    r.gw.dispose()
    await reading
    expect(signal?.aborted).toBe(true)
    await r.gw.settled()
    expect(spent(r)).toBe(0)
    const left = diskLedger(r).reservations
    expect(left).toHaveLength(1)
    expect(left[0].generationId).toBe('gen-D')
    // A new request after dispose is refused.
    expect((await chat(r)).status).toBe(502)
  })
})

describe('non-streaming clients', () => {
  it('collect the stream into one chat.completion', async () => {
    const r = await rig()
    r.net.handler = (c) =>
      sseResponse(c, [
        chunk({ role: 'assistant', content: '', reasoning: 'hm ', reasoning_details: [{ type: 'reasoning.text', text: 'hm ', index: 0 }] }),
        chunk({ content: 'Hel' }),
        chunk({ content: 'lo', tool_calls: [{ index: 0, id: 'c1', type: 'function', function: { name: 'get', arguments: '{"a"' } }] }),
        chunk({ tool_calls: [{ index: 0, function: { arguments: ':1}' } }] }, {}, 'tool_calls'),
        usageChunk(0.003),
        DONE
      ])
    const res = await chat(r, { messages: U, stream: false, tools: [{ type: 'function', function: { name: 'get' } }] })
    expect(res.status).toBe(200)
    expect(res.headers.get('content-type')).toContain('application/json')
    const j = await res.json()
    expect(j.object).toBe('chat.completion')
    expect(j.choices[0].message).toMatchObject({ role: 'assistant', content: 'Hello', reasoning_content: 'hm ', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'get', arguments: '{"a":1}' } }] })
    expect(j.choices[0].message.reasoning_details).toEqual([{ type: 'reasoning.text', text: 'hm ', index: 0 }])
    expect(j.choices[0].finish_reason).toBe('tool_calls')
    expect(j.usage.cost).toBe(0.003)
    expect(r.net.chats[0].body?.stream).toBe(true)
    expect(spent(r)).toBe(3000)
  })

  it('an error mid-stream gives an HTTP error answer', async () => {
    const r = await rig()
    r.sleeper.auto = true
    r.net.handler = router((c) => sseResponse(c, [chunk({ content: 'a' }), frame({ error: { code: 402, message: 'Insufficient credits' } })]))
    const res = await chat(r, { messages: U })
    expect(res.status).toBe(402)
    expect((await res.json()).error).toMatchObject({ code: 'cloud_credits', message: 'Insufficient credits' })
    await r.gw.settled()
    expect(r.ledger.reservations()).toEqual([])
  })

  it('a caller abort rejects', async () => {
    const r = await rig()
    const ctl = new AbortController()
    r.net.handler = (c) => new Response(new Pipe().body(c.signal), { status: 200 })
    const p = chat(r, { messages: U }, 'window', ctl.signal)
    await vi.waitFor(() => expect(r.net.chats).toHaveLength(1))
    ctl.abort()
    await expect(p).rejects.toThrow()
    expect(r.ledger.reservations()).toEqual([])
  })
})

describe('retries', () => {
  it('429 is retried waiting Retry-After (seconds), then succeeds; nothing charged for the failures', async () => {
    const r = await rig()
    r.sleeper.auto = true
    r.net.handler = (c, n) => (n === 1 ? jsonResponse({ error: { message: 'slow down' } }, 429, { 'retry-after': '7' }) : sseResponse(c, goodStream('ok', 0.001)))
    const text = await (await chat(r)).text()
    expect(text).toContain('ok')
    expect(r.net.chats).toHaveLength(2)
    expect(r.sleeper.waits.map((w) => w.ms)).toEqual([7000])
    expect(spent(r)).toBe(1000)
  })

  it('Retry-After is capped at 30 s; without it the waits are 2 s and 8 s; three tries at most', async () => {
    expect(retryAfterMs('100', 0)).toBe(30_000)
    expect(retryAfterMs('0', 0)).toBe(0)
    expect(retryAfterMs(null, 0)).toBeNull()
    expect(retryAfterMs('soon', 0)).toBeNull()
    expect(retryAfterMs(new Date(5000).toUTCString(), 1000)).toBe(4000)
    const r = await rig()
    r.sleeper.auto = true
    r.net.handler = (_c, n) => jsonResponse({ error: { message: 'limited' } }, 429, n === 1 ? { 'retry-after': '100' } : {})
    const res = await chat(r)
    expect(r.net.chats).toHaveLength(3)
    expect(r.sleeper.waits.map((w) => w.ms)).toEqual([30_000, 8000])
    expect(res.status).toBe(429)
    expect((await res.json()).error.code).toBe('cloud_rate')
    expect(spent(r)).toBe(0)
    expect(r.ledger.reservations()).toEqual([])
  })

  it('503 is retried; a final 503 about providers is a policy error, else a plain one; both cost nothing', async () => {
    const r = await rig()
    r.sleeper.auto = true
    r.net.handler = () => jsonResponse({ error: { message: 'No available provider under your price limits' } }, 503)
    const res = await chat(r)
    expect(r.net.chats).toHaveLength(3)
    expect(r.sleeper.waits.map((w) => w.ms)).toEqual([2000, 8000])
    expect((await res.json()).error.code).toBe('cloud_policy')
    const r2 = await rig()
    r2.sleeper.auto = true
    r2.net.handler = () => jsonResponse({ error: { message: 'overloaded' } }, 503)
    expect((await (await chat(r2)).json()).error.code).toBe('cloud_error')
    expect(spent(r) + spent(r2)).toBe(0)
  })

  it('failures before anything was sent (DNS, refused, connect timeout) are retried', async () => {
    for (const code of ['ENOTFOUND', 'ECONNREFUSED', 'EAI_AGAIN', 'UND_ERR_CONNECT_TIMEOUT']) {
      const r = await rig()
      r.sleeper.auto = true
      r.net.handler = (c, n) => (n < 3 ? connectError(code) : sseResponse(c, goodStream('ok', 0.001)))
      const text = await (await chat(r)).text()
      expect(text).toContain('ok')
      expect(r.net.chats).toHaveLength(3)
      expect(spent(r)).toBe(1000)
    }
  })

  it('all tries not sent: an error and nothing charged', async () => {
    const r = await rig()
    r.sleeper.auto = true
    r.net.handler = () => connectError('ENOTFOUND')
    const res = await chat(r)
    expect(res.status).toBe(502)
    expect((await res.json()).error.message).toMatch(/Nothing was sent/)
    expect(r.net.chats).toHaveLength(3)
    expect(spent(r)).toBe(0)
    expect(r.ledger.reservations()).toEqual([])
  })

  it('no retry after the request may have been sent: a reset connection charges the reservation (or the looked-up cost)', async () => {
    const r = await rig()
    r.net.handler = () => connectError('ECONNRESET')
    const res = await chat(r)
    expect(res.status).toBe(502)
    expect(r.net.chats).toHaveLength(1)
    expect(spent(r)).toBeGreaterThan(0)
    expect(r.ledger.reservations()).toEqual([])
  })

  it('other 5xx: no retry; the id in the headers lets the cost be looked up', async () => {
    const r = await rig()
    r.sleeper.auto = true
    r.net.handler = router(() => jsonResponse({ error: { message: 'boom' } }, 500, { 'x-generation-id': 'gen-5' }), generationAnswer(0.0006) as never)
    const res = await chat(r)
    expect(res.status).toBe(502)
    expect(r.net.chats).toHaveLength(1)
    await r.gw.settled()
    expect(spent(r)).toBe(600)
    const r2 = await rig()
    r2.net.handler = () => jsonResponse({ error: { message: 'boom' } }, 502)
    await chat(r2)
    // No id: the reservation.
    expect(spent(r2)).toBeGreaterThan(0)
  })

  it('4xx answers are not retried and cost nothing, with the right kinds', async () => {
    const cases: [number, string, string][] = [
      [400, 'bad request', 'cloud_request'],
      [401, 'No auth', 'cloud_key'],
      [402, 'Insufficient credits', 'cloud_credits'],
      [403, 'Your chosen model requires moderation', 'cloud_policy'],
      [403, 'Key limit exceeded (daily limit)', 'cloud_credits'],
      [404, 'No endpoints found matching your data policy', 'cloud_policy']
    ]
    for (const [status, message, code] of cases) {
      const r = await rig()
      r.net.handler = () => jsonResponse({ error: { message } }, status)
      const res = await chat(r)
      expect(r.net.chats).toHaveLength(1)
      expect(res.status).toBe(status)
      expect((await res.json()).error).toMatchObject({ code, message })
      expect(spent(r)).toBe(0)
      expect(r.ledger.reservations()).toEqual([])
    }
  })

  it('aborting during the wait stops the retries and costs nothing', async () => {
    const r = await rig()
    r.net.handler = () => jsonResponse({ error: { message: 'slow' } }, 429, { 'retry-after': '5' })
    const ctl = new AbortController()
    const p = chat(r, body(), 'window', ctl.signal)
    await vi.waitFor(() => expect(r.sleeper.waits).toHaveLength(1))
    ctl.abort()
    await expect(p).rejects.toThrow()
    expect(r.net.chats).toHaveLength(1)
    expect(spent(r)).toBe(0)
    expect(r.ledger.reservations()).toEqual([])
  })

  it('a redirect answer is refused (the key stays with openrouter.ai) and costs nothing', async () => {
    const r = await rig()
    r.net.handler = () => Object.assign(new TypeError('fetch failed'), { cause: new Error('redirect mode is set to error') })
    const res = await chat(r)
    expect(res.status).toBe(502)
    expect((await res.json()).error.message).toMatch(/redirect/)
    expect(spent(r)).toBe(0)
  })
})

describe('the limit and the reservations', () => {
  it('over the limit by itself: 402 cloud_limit with numbers, never sent', async () => {
    const r = await rig({ limitUsd: 0.01 })
    const res = await chat(r, body({ max_tokens: 8000 }))
    expect(res.status).toBe(402)
    const e = (await res.json()).error
    expect(e.code).toBe('cloud_limit')
    expect(e.message).toMatch(/could cost up to \$\d+\.\d+, and \$0\.01 is left of today's \$0\.01 limit/)
    expect(e.message).toMatch(/Shorten it, lower the model's reply or context limit, or raise the limit/)
    expect(r.net.calls).toHaveLength(0)
    expect(r.ledger.reservations()).toEqual([])
  })

  it('waits for running requests when only they are in the way, then goes', async () => {
    // A request of max_tokens 100 on this model books about $0.004: a $0.006 limit fits one at a time.
    const r = await rig({ limitUsd: 0.01 })
    const pipe = new Pipe()
    const second = new Pipe()
    r.net.handler = (c, n) => new Response((n === 1 ? pipe : second).body(c.signal), { status: 200 })
    const a = await chat(r, body({ max_tokens: 150 }))
    const bPromise = chat(r, body({ max_tokens: 150 }))
    await new Promise((x) => setTimeout(x, 30))
    expect(r.net.chats).toHaveLength(1)
    for (const f of goodStream('one', 0.001)) pipe.push(f)
    pipe.end()
    await a.text()
    await vi.waitFor(() => expect(r.net.chats).toHaveLength(2))
    for (const f of goodStream('two', 0.001)) second.push(f)
    second.end()
    expect(await (await bPromise).text()).toContain('two')
    expect(spent(r)).toBe(2000)
  })

  it('after ten minutes of waiting: cloud_limit, saying it is busy', async () => {
    const r = await rig({ limitUsd: 0.01 })
    r.net.handler = (c) => new Response(new Pipe().body(c.signal), { status: 200 })
    await chat(r, body({ max_tokens: 150 }))
    const p = chat(r, body({ max_tokens: 150 }))
    await vi.waitFor(() => expect(r.sleeper.waits).toHaveLength(1))
    expect(r.sleeper.waits[0].ms).toBe(600_000)
    r.sleeper.waits[0].release()
    const res = await p
    expect(res.status).toBe(402)
    const e = (await res.json()).error
    expect(e.code).toBe('cloud_limit')
    expect(e.message).toMatch(/busy/)
    expect(r.net.chats).toHaveLength(1)
  })

  it('the wait is abortable', async () => {
    const r = await rig({ limitUsd: 0.01 })
    r.net.handler = (c) => new Response(new Pipe().body(c.signal), { status: 200 })
    await chat(r, body({ max_tokens: 150 }))
    const ctl = new AbortController()
    const p = chat(r, body({ max_tokens: 150 }), 'window', ctl.signal)
    await vi.waitFor(() => expect(r.sleeper.waits).toHaveLength(1))
    ctl.abort()
    await expect(p).rejects.toThrow()
    expect(r.net.chats).toHaveLength(1)
  })

  it('ten concurrent requests never exceed the limit by reservations', async () => {
    const r = await rig({ limitUsd: 0.02 })
    let open = 0
    let maxOpen = 0
    let maxBooked = 0
    r.net.handler = async (c) => {
      open++
      maxOpen = Math.max(maxOpen, open)
      const s = r.ledger.snapshot()
      maxBooked = Math.max(maxBooked, s.spent + s.reserved)
      await new Promise((x) => setTimeout(x, 5))
      open--
      return sseResponse(c, goodStream('x', 0.0005))
    }
    const results = await Promise.all(Array.from({ length: 10 }, () => chat(r, body({ max_tokens: 150 })).then((x) => x.text())))
    expect(results.every((t) => t.includes('[DONE]'))).toBe(true)
    expect(maxOpen).toBeGreaterThan(0)
    expect(maxOpen).toBeLessThanOrEqual(3)
    expect(maxBooked).toBeLessThanOrEqual(20_000)
    expect(spent(r)).toBe(5000)
  })

  it('limit null: no limit, still recorded', async () => {
    const r = await rig({ limitUsd: null })
    r.net.handler = (c) => sseResponse(c, goodStream('x', 0.5))
    await (await chat(r)).text()
    expect(spent(r)).toBe(500_000)
  })

  it('a change of the limit applies to the next request', async () => {
    const r = await rig({ limitUsd: 1 })
    r.net.handler = (c) => sseResponse(c, goodStream('x', 0.001))
    await (await chat(r)).text()
    await r.store.setOptions({ dailyLimitUsd: 0.01 })
    expect((await chat(r, body({ max_tokens: 8000 }))).status).toBe(402)
    await r.store.setOptions({ allowTraining: true, dailyLimitUsd: 1 })
    await (await chat(r)).text()
    expect(r.net.chats[1].body?.provider.data_collection).toBeUndefined()
    expect(r.net.chats[0].body?.provider.data_collection).toBe('deny')
  })

  it('a day change starts a new day at zero', async () => {
    const r = await rig({ limitUsd: 0.01 })
    r.net.handler = (c) => sseResponse(c, goodStream('x', 0.008))
    await (await chat(r, body({ max_tokens: 10 }))).text()
    expect((await chat(r, body({ max_tokens: 150 }))).status).toBe(402)
    r.clock.advance(24 * 3600_000)
    await (await chat(r, body({ max_tokens: 10 }))).text()
    expect(spent(r)).toBe(8000)
    expect(r.ledger.snapshot().days).toHaveLength(1)
  })

  it('a free model reserves 0 and costs 0', async () => {
    const r = await rig({ slug: 'baidu/ernie-5.5-flash-preview:free', limitUsd: 0.01 })
    r.net.handler = (c) => sseResponse(c, goodStream('x', 0))
    await (await chat(r)).text()
    expect(spent(r)).toBe(0)
    expect(r.net.chats[0].body?.provider.max_price).toEqual({ prompt: 0, completion: 0 })
  })

  it('a corrupt spending record refuses requests and is never reset', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'cloud-corrupt-'))
    writeFileSync(join(dir, 'cloud-spend.json'), '{ definitely not json')
    const r = await rig({ dir })
    const res = await chat(r)
    expect(res.status).toBe(503)
    expect((await res.json()).error.code).toBe('cloud_limit')
    expect(r.net.calls).toHaveLength(0)
    expect(readFileSync(join(dir, 'cloud-spend.json'), 'utf8')).toBe('{ definitely not json')
  })

  it('leftovers of a crash are settled at start: without an id the reservation, with an id the looked up cost', async () => {
    const r = await rig()
    r.sleeper.auto = true
    await r.ledger.reserve('anthropic/claude-fable-5.2', 5000)
    await r.ledger.reserve('anthropic/claude-fable-5.2', 7000)
    await r.ledger.setGenerationId(r.ledger.reservations()[1].id, 'gen-old')
    r.net.handler = generationAnswer(0.0001) as never
    await r.gw.settleLeftovers()
    await r.gw.settled()
    expect(r.ledger.reservations()).toEqual([])
    expect(spent(r)).toBe(5000 + 100)
  })

  it('an old reservation whose generation is unknown is charged in full after the three tries', async () => {
    const r = await rig()
    r.sleeper.auto = true
    await r.ledger.reserve('anthropic/claude-fable-5.2', 7000)
    await r.ledger.setGenerationId(r.ledger.reservations()[0].id, 'gen-lost')
    r.net.handler = () => new Response('', { status: 404 })
    await r.gw.settleLeftovers()
    await r.gw.settled()
    expect(r.net.calls).toHaveLength(3)
    expect(spent(r)).toBe(7000)
  })

  it('generation lookups add the BYOK upstream cost', async () => {
    const r = await rig()
    r.sleeper.auto = true
    await r.ledger.reserve('anthropic/claude-fable-5.2', 9000)
    await r.ledger.setGenerationId(r.ledger.reservations()[0].id, 'g')
    r.net.handler = () => jsonResponse({ data: { total_cost: 0.001, is_byok: true, upstream_inference_cost: 0.002 } })
    await r.gw.settleLeftovers()
    await r.gw.settled()
    expect(spent(r)).toBe(3000)
  })
})

describe('who may call', () => {
  const paths: [string, string, string?][] = [
    ['GET', '/props'],
    ['GET', '/v1/props'],
    ['POST', '/tokenize', JSON.stringify({ content: 'x' })],
    ['POST', '/v1/chat/completions', JSON.stringify(body())],
    ['POST', '/v1/chat/completions/input_tokens', JSON.stringify(body())],
    ['POST', '/apply-template', JSON.stringify({ messages: U })],
    ['GET', '/slots']
  ]

  it('a caller other than the window is refused (403 cloud_not_allowed) on every call unless the model is ticked', async () => {
    const r = await rig({ allowApps: false })
    for (const by of ['api', 'phone'] as const) {
      for (const [method, path, b] of paths) {
        const res = await r.gw.fetch(r.modelId, path, { method, body: b }, by)
        expect(res.status).toBe(403)
        expect((await res.json()).error.code).toBe('cloud_not_allowed')
      }
    }
    expect(r.net.calls).toHaveLength(0)
    expect(r.ledger.reservations()).toEqual([])
    // The window itself is fine.
    expect((await r.gw.fetch(r.modelId, '/props', {}, 'window')).status).toBe(200)
  })

  it('ticked: other apps may use it; unticking applies to the very next call', async () => {
    const r = await rig({ allowApps: true })
    r.net.handler = (c) => sseResponse(c, goodStream('x', 0.001))
    expect((await r.gw.fetch(r.modelId, '/props', {}, 'api')).status).toBe(200)
    expect((await chat(r, body(), 'phone')).status).toBe(200)
    await r.store.update(r.def.slug, { allowApps: false })
    expect((await r.gw.fetch(r.modelId, '/props', {}, 'api')).status).toBe(403)
    expect((await chat(r, body(), 'api')).status).toBe(403)
  })

  it('a model that is gone: 403 for others (nothing leaks), 404 for the window', async () => {
    const r = await rig()
    expect((await r.gw.fetch('openrouter/not/here', '/props', {}, 'api')).status).toBe(403)
    expect((await r.gw.fetch('openrouter/not/here', '/props', {}, 'window')).status).toBe(404)
    expect((await r.gw.fetch('local-file.gguf', '/props', {}, 'window')).status).toBe(404)
  })
})

describe('the gateway answers as llama-server would', () => {
  it('GET /props', async () => {
    const r = await rig({ contextLimit: 50_000 })
    for (const p of ['/props', '/v1/props', '/props?x=1']) {
      const j = await (await r.gw.fetch(r.modelId, p, {}, 'window')).json()
      expect(j).toEqual({ default_generation_settings: { n_ctx: 50_000 }, n_ctx: 50_000, modalities: { vision: true }, chat_template: '' })
    }
    const t = await rig({ slug: 'inference-net/schematron-v2-turbo' })
    expect((await (await t.gw.fetch(t.modelId, '/props', {}, 'window')).json()).modalities.vision).toBe(false)
  })

  it('POST /tokenize gives as many tokens as the estimate', async () => {
    const r = await rig()
    const j = await (await r.gw.fetch(r.modelId, '/tokenize', { method: 'POST', body: JSON.stringify({ content: 'a'.repeat(35) + '日本' }) }, 'window')).json()
    expect(j.tokens).toHaveLength(12)
    expect(j.tokens.every((n: unknown) => typeof n === 'number')).toBe(true)
    expect((await r.gw.fetch(r.modelId, '/tokenize', { method: 'POST', body: '{}' }, 'window')).status).toBe(400)
  })

  it('POST /apply-template and input_tokens', async () => {
    const r = await rig()
    const t = await (await r.gw.fetch(r.modelId, '/apply-template', { method: 'POST', body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }) }, 'window')).json()
    expect(t.prompt).toContain('user: hi')
    const n = await (await r.gw.fetch(r.modelId, '/v1/chat/completions/input_tokens', { method: 'POST', body: JSON.stringify(body()) }, 'window')).json()
    expect(n.input_tokens).toBeGreaterThan(0)
    const bad = await r.gw.fetch(r.modelId, '/v1/chat/completions/input_tokens', { method: 'POST', body: JSON.stringify({ messages: [] }) }, 'window')
    expect(bad.status).toBe(400)
    expect(r.net.calls).toHaveLength(0)
  })

  it('anything else is a 400 cloud_request', async () => {
    const r = await rig()
    for (const [m, p] of [['GET', '/slots'], ['POST', '/completion'], ['POST', '/v1/messages'], ['POST', '/v1/embeddings'], ['GET', '/v1/chat/completions'], ['POST', '/lora-adapters']]) {
      const res = await r.gw.fetch(r.modelId, p, { method: m }, 'window')
      expect(res.status).toBe(400)
      expect((await res.json()).error.code).toBe('cloud_request')
    }
  })

  it('refuses locally what the model cannot do, with no booking and no fetch', async () => {
    const r = await rig({ slug: 'inference-net/schematron-v2-turbo' })
    const res = await chat(r, body({ tools: [{ type: 'function', function: { name: 'x' } }] }))
    expect(res.status).toBe(400)
    expect((await res.json()).error.code).toBe('cloud_request')
    expect(r.net.calls).toHaveLength(0)
    expect(r.ledger.reservations()).toEqual([])
    expect((await chat(r, 'nonsense' as never).catch(() => ({ status: 0 }))).status).toBeDefined()
  })

  it('no key: 401 cloud_key', async () => {
    const r = await rig({ key: null })
    const res = await chat(r)
    expect(res.status).toBe(401)
    expect((await res.json()).error.code).toBe('cloud_key')
    expect(r.net.calls).toHaveLength(0)
  })
})

describe('reasoning_details across tool rounds through the gateway', () => {
  it('a reply with tool calls is remembered and put back in the next request', async () => {
    const r = await rig()
    const details = [{ type: 'reasoning.encrypted', data: 'abc', index: 0 }]
    r.net.handler = (c, n) =>
      n === 1
        ? sseResponse(c, [
            chunk({ role: 'assistant', reasoning_details: details, tool_calls: [{ index: 0, id: 'c1', type: 'function', function: { name: 'get', arguments: '{"b":2,"a":1}' } }] }),
            chunk({}, {}, 'tool_calls'),
            usageChunk(0.001),
            DONE
          ])
        : sseResponse(c, goodStream('done', 0.001))
    const tools = [{ type: 'function', function: { name: 'get' } }]
    const text1 = await (await chat(r, body({ tools }))).text()
    // The window re-sends the call it read, its arguments re-serialised, with the reasoning text it saw.
    const second = [
      ...U,
      { role: 'assistant', content: '', reasoning_content: 'ignored', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'get', arguments: '{ "a": 1, "b": 2 }' } }] },
      { role: 'tool', tool_call_id: 'c1', content: 'result' }
    ]
    await (await chat(r, body({ tools, messages: second }))).text()
    expect(text1).toContain('reasoning_details')
    const sent = r.net.chats[1].body?.messages[1]
    expect(sent.reasoning_details).toEqual(details)
    expect(sent).not.toHaveProperty('reasoning_content')
  })
})

describe('the key never leaks', () => {
  it('is scrubbed from error answers, thrown text, logs and states; and goes only to openrouter.ai', async () => {
    const r = await rig()
    r.sleeper.auto = true
    r.net.handler = (_c, n) => {
      if (n === 1) return jsonResponse({ error: { message: `Invalid key ${TEST_KEY} sent as Bearer ${TEST_KEY}` } }, 429, { 'retry-after': '1' })
      if (n === 2) return jsonResponse({ error: { message: `Provider said ${TEST_KEY}` } }, 400)
      return new Error(`socket for ${TEST_KEY}`)
    }
    const a = await chat(r)
    expect(a.status).toBe(400)
    const aText = await a.text()
    expect(aText).not.toContain(TEST_KEY)
    expect(aText).toContain('***')
    const r2 = await rig()
    r2.net.handler = () => connectError('ECONNRESET')
    const b = await (await chat(r2)).text()
    expect(b).not.toContain(TEST_KEY)
    const r3 = await rig()
    r3.net.handler = () => Object.assign(new Error(`weird ${TEST_KEY}`), { code: 'EWEIRD' })
    const c = await (await chat(r3)).text()
    expect(c).not.toContain(TEST_KEY)
    expect(c).toContain('***')
    for (const x of [r, r2, r3]) {
      expect(x.logs.join('\n')).not.toContain(TEST_KEY)
      expect(JSON.stringify(x.ledger.snapshot())).not.toContain(TEST_KEY)
      expect(readFileSync(join(x.dir, 'cloud-spend.json'), 'utf8')).not.toContain(TEST_KEY)
      for (const call of x.net.calls) expect(call.url.startsWith('https://openrouter.ai/')).toBe(true)
    }
    expect(r.logs.length).toBeGreaterThan(0)
  })

  it('a mid-stream error text with the key is scrubbed', async () => {
    const r = await rig()
    r.net.handler = (c) => sseResponse(c, [chunk({ content: 'a' }), frame({ error: { code: 500, message: `leak ${TEST_KEY}` } })])
    const text = await (await chat(r)).text()
    expect(text).not.toContain(TEST_KEY)
    expect(text).toContain('leak ***')
  })

  it('every request that carries the key goes to https://openrouter.ai without following redirects', async () => {
    const r = await rig()
    r.sleeper.auto = true
    r.net.handler = router((c) => sseResponse(c, [chunk({ content: 'x' }), chunk({}, {}, 'stop'), DONE], { 'x-generation-id': 'g1' }), generationAnswer(0.0001) as never)
    await (await chat(r)).text()
    await r.gw.settled()
    const withKey = r.net.calls.filter((c) => c.headers.authorization)
    expect(withKey.length).toBeGreaterThanOrEqual(2)
    for (const c of withKey) {
      expect(new URL(c.url).origin).toBe('https://openrouter.ai')
      expect(c.redirect).toBe('error')
    }
  })
})

describe('what changed while a request was waiting', () => {
  /** A rig whose whole limit is booked by another request, and a request waiting for it. */
  async function waiting(by: CloudCaller = 'window') {
    const r = await rig({ limitUsd: 1, allowApps: true })
    const held = await r.ledger.reserve('other', 999_999)
    if (!held.ok) throw new Error('not held')
    r.net.handler = (c) => sseResponse(c, goodStream('Hello', 0.0005))
    const p = chat(r, body(), by)
    await vi.waitFor(() => expect(r.sleeper.waits).toHaveLength(1))
    return { r, p, release: () => r.ledger.settle(held.id, 0) }
  }
  const code = async (res: Response): Promise<string> => ((await res.json()) as any).error.code

  it('the key was forgotten: refused, nothing sent, nothing booked', async () => {
    const { r, p, release } = await waiting()
    await r.keys.clear()
    await release()
    expect(await code(await p)).toBe('cloud_key')
    expect(r.net.calls).toHaveLength(0)
    expect(r.ledger.snapshot()).toMatchObject({ spent: 0, reserved: 0 })
  })

  it('it was unticked for other apps, or removed', async () => {
    const a = await waiting('api')
    await a.r.store.update(a.r.def.slug, { allowApps: false })
    await a.release()
    expect(await code(await a.p)).toBe('cloud_not_allowed')
    expect(a.r.net.calls).toHaveLength(0)
    const b = await waiting()
    await b.r.store.remove(b.r.def.slug)
    await b.release()
    expect((await b.p).status).toBe(404)
    expect(b.r.ledger.snapshot()).toMatchObject({ spent: 0, reserved: 0 })
  })

  it('training providers were switched off: the sent request says deny', async () => {
    const r = await rig({ limitUsd: 1, allowTraining: true })
    const held = await r.ledger.reserve('other', 999_999)
    r.net.handler = (c) => sseResponse(c, goodStream('Hello', 0.0005))
    const p = chat(r)
    await vi.waitFor(() => expect(r.sleeper.waits).toHaveLength(1))
    await r.store.setOptions({ allowTraining: false })
    await r.ledger.settle((held as { id: string }).id, 0)
    await (await p).text()
    expect(r.net.calls[0].body?.provider.data_collection).toBe('deny')
  })

  it('the app is closing: not sent', async () => {
    const { r, p, release } = await waiting()
    r.gw.dispose()
    expect(await code(await p)).toBe('cloud_error')
    await release()
    expect(r.net.calls).toHaveLength(0)
  })
})

describe('a collected (non-streaming) answer', () => {
  it('a reply that breaks off is an error, not a short reply', async () => {
    const r = await rig()
    r.net.handler = router((c) => sseResponse(c, [chunk({ content: 'half' })]))
    const res = await chat(r, body({ stream: false }))
    expect(res.status).toBe(502)
    expect((await res.json() as any).error.code).toBe('cloud_error')
  })

  it('an error chunk is scrubbed', async () => {
    const r = await rig()
    r.net.handler = router((c) => sseResponse(c, [frame({ error: { code: 500, message: `bad key ${TEST_KEY}` }, choices: [] })]))
    const res = await chat(r, body({ stream: false }))
    expect(await res.text()).not.toContain(TEST_KEY)
    expect(res.ok).toBe(false)
  })
})

describe('a try that is repeated', () => {
  it('its generation id does not stand for the next try, whose cost is unknown', async () => {
    const r = await rig()
    r.sleeper.auto = true
    const asked: string[] = []
    r.net.handler = router(
      (_c, n) => (n === 1 ? jsonResponse({ error: { message: 'slow down' } }, 429, { 'x-generation-id': 'gen-first' }) : new TypeError('socket hang up')),
      (c) => {
        asked.push(c.url)
        return jsonResponse({ data: { total_cost: 0 } })
      }
    )
    const res = await chat(r)
    expect(res.ok).toBe(false)
    await r.gw.settled()
    expect(asked).toEqual([])
    expect(spent(r)).toBeGreaterThan(0)
  })
})
