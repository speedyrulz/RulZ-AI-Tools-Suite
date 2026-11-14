import { mkdtempSync, rmSync } from 'node:fs'
import { createServer, type IncomingMessage, type Server } from 'node:http'
import type { AddressInfo } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import type { RunEvent, WireMessage } from '@shared/types'
import { AppRunner } from '../src/main/project/runApp'
import { friendlyServerError, isNetworkError, RulzClient, ServerError } from '../src/main/rulz/client'
import { jsonifyToolCallArgs, validToolCallArgs } from '../src/main/rulz/toolArgs'

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = []
  for await (const c of req) chunks.push(c as Buffer)
  return Buffer.concat(chunks).toString('utf8')
}

const servers: Server[] = []
afterAll(async () => {
  await Promise.all(servers.map((s) => new Promise((r) => s.close(r))))
})

/** A server answering like llama-server behind RulZ LLM Suite (which passes the body and the stream through). */
async function startServer(opts: { props?: Record<string, unknown>; models?: Record<string, unknown>[] } = {}) {
  const requests: { path: string; body?: Record<string, unknown> }[] = []
  const server = createServer(async (req, res) => {
    const url = new URL(req.url ?? '/', 'http://x')
    const json = (body: unknown): void => void res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(body))
    if (url.pathname === '/props') {
      requests.push({ path: url.pathname })
      return json(opts.props ?? { chat_template: 'plain' })
    }
    if (url.pathname === '/v1/models') return json({ object: 'list', data: opts.models ?? [] })
    if (url.pathname === '/v1/chat/completions') {
      const body = JSON.parse(await readBody(req)) as Record<string, unknown>
      requests.push({ path: url.pathname, body })
      res.writeHead(200, { 'Content-Type': 'text/event-stream' })
      const base = { created: 1, id: 'chatcmpl-1', model: 'm', system_fingerprint: 'b1', object: 'chat.completion.chunk' }
      const send = (o: object): void => void res.write(`data: ${JSON.stringify({ ...base, ...o })}\n\n`)
      const role = { choices: [{ finish_reason: null, index: 0, delta: { role: 'assistant', content: null } }] }
      // llama-server with return_progress: a role-only delta per batch, prompt_progress beside it
      // (processed counts the cached tokens too).
      if (body.return_progress === true) {
        for (const processed of [3048, 5048, 6000]) send({ ...role, prompt_progress: { total: 6000, cache: 1000, processed, time_ms: 5 } })
      }
      send(role)
      send({ choices: [{ finish_reason: null, index: 0, delta: { content: 'Hello.' } }] })
      send({ choices: [{ finish_reason: 'stop', index: 0, delta: {} }], timings: { cache_n: 1000, prompt_n: 5000, predicted_n: 2, predicted_ms: 10 } })
      res.end('data: [DONE]\n\n')
      return
    }
    res.writeHead(404).end()
  })
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r))
  servers.push(server)
  return { url: `http://127.0.0.1:${(server.address() as AddressInfo).port}`, requests }
}

describe('prompt progress', () => {
  it('asks for it and reports llama-server progress chunks without the cached part', async () => {
    const s = await startServer()
    const client = new RulzClient(() => ({ url: s.url, apiKey: '' }))
    const progress: [number, number][] = []
    const turn = await client.streamTurn({ model: 'm', messages: [{ role: 'user', content: 'hi' }] }, new AbortController().signal, {
      onProgress: (done, total) => progress.push([done, total])
    })
    expect(s.requests[0].body).toMatchObject({ stream: true, return_progress: true })
    expect(progress).toEqual([
      [2048, 5000],
      [4048, 5000],
      [5000, 5000]
    ])
    expect(turn.content).toBe('Hello.')
    expect(turn.usage).toEqual({ prompt: 6000, completion: 2 })
  })
})

describe('lost-connection message', () => {
  const lost = /lost the connection to RulZ LLM Suite/

  it('is given for real network failures only', async () => {
    const refused = Object.assign(new TypeError('fetch failed'), { cause: Object.assign(new Error('connect ECONNREFUSED 127.0.0.1:9'), { code: 'ECONNREFUSED' }) })
    expect(friendlyServerError(refused)).toMatch(lost)
    expect(friendlyServerError(new TypeError('fetch failed'))).toMatch(lost)
    // A stream cut off mid-reply.
    expect(friendlyServerError(Object.assign(new TypeError('terminated'), { cause: Object.assign(new Error('other side closed'), { code: 'UND_ERR_SOCKET' }) }))).toMatch(lost)
    expect(friendlyServerError(Object.assign(new Error('read ECONNRESET'), { code: 'ECONNRESET' }))).toMatch(lost)
    expect(friendlyServerError(Object.assign(new Error('x'), { cause: { cause: { code: 'ETIMEDOUT' } } }))).toMatch(lost)
    // The real thing: nothing listens on port 9.
    const real = await fetch('http://127.0.0.1:9/v1/models').catch((e: unknown) => e)
    expect(isNetworkError(real)).toBe(true)
    expect(friendlyServerError(real)).toMatch(lost)
  })

  it('keeps the message of mistakes in the code', () => {
    const bug = new TypeError("Cannot read properties of undefined (reading 'content')")
    expect(friendlyServerError(bug)).toBe("Something went wrong: Cannot read properties of undefined (reading 'content')")
    const badArg = Object.assign(new TypeError('invalid header value'), { cause: { code: 'UND_ERR_INVALID_ARG' } })
    expect(friendlyServerError(badArg)).toBe('Something went wrong: invalid header value')
    expect(friendlyServerError(Object.assign(new Error('x'), { name: 'AbortError' }))).toBe('Stopped.')
    expect(friendlyServerError(new ServerError('bad key', 401))).toMatch(/rejected the API key/)
    expect(friendlyServerError(undefined)).toBe('Something went wrong: undefined')
  })
})

describe('cached model props', () => {
  it('are kept while the context length is unknown on either side', async () => {
    const models = [{ id: 'm', owned_by: 'rulz', path: 'C:\\m.gguf', state: 'loaded', loaded_context_length: 32768 }]
    const s = await startServer({ props: { chat_template: '<parameter= args_value | string' }, models })
    const client = new RulzClient(() => ({ url: s.url, apiKey: '' }))
    expect((await client.modelProps('m')).nCtx).toBeUndefined()
    for (let i = 0; i < 3; i++) await client.listModels()
    await client.modelProps('m')
    expect(s.requests.filter((r) => r.path === '/props')).toHaveLength(1)
  })

  it('are kept while the list shows no loaded context, and dropped when it really changed', async () => {
    const models: Record<string, unknown>[] = [{ id: 'm', owned_by: 'rulz', path: 'C:\\m.gguf', state: 'not-loaded' }]
    const props = { chat_template: 't', default_generation_settings: { n_ctx: 65536 } }
    const s = await startServer({ props, models })
    const client = new RulzClient(() => ({ url: s.url, apiKey: '' }))
    await client.modelProps('m')
    await client.listModels()
    await client.modelProps('m')
    expect(s.requests.filter((r) => r.path === '/props')).toHaveLength(1)
    // Same context once loaded: still kept.
    Object.assign(models[0], { state: 'loaded', loaded_context_length: 65536 })
    await client.listModels()
    await client.modelProps('m')
    expect(s.requests.filter((r) => r.path === '/props')).toHaveLength(1)
    // Reloaded with another context: fetched again.
    models[0].loaded_context_length = 16384
    props.default_generation_settings.n_ctx = 16384
    await client.listModels()
    expect((await client.modelProps('m')).nCtx).toBe(16384)
    expect(s.requests.filter((r) => r.path === '/props')).toHaveLength(2)
  })
})

describe('tool-call argument fixes are worked out once per call', () => {
  const call = (id: string, args: string) => ({ id, type: 'function' as const, function: { name: 'write_file', arguments: args } })
  it('gives the same results, reusing them on later turns', () => {
    const repaired = call('a', "{'deep': True}")
    const good = call('b', '{"n":2}')
    const msgs: WireMessage[] = [
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: '', tool_calls: [repaired, good] }
    ]
    const first = validToolCallArgs(msgs)
    const second = validToolCallArgs(msgs)
    expect(JSON.parse(first[1].tool_calls![0].function.arguments)).toEqual({ deep: true })
    expect(second[1].tool_calls![0]).toBe(first[1].tool_calls![0])
    expect(second[1].tool_calls![1]).toBe(good)

    const j1 = jsonifyToolCallArgs(first)
    const j2 = jsonifyToolCallArgs(second)
    expect(JSON.parse(j1[1].tool_calls![0].function.arguments)).toEqual({ deep: 'true' })
    expect(JSON.parse(j1[1].tool_calls![1].function.arguments)).toEqual({ n: '2' })
    expect(j2[1].tool_calls![0]).toBe(j1[1].tool_calls![0])
    expect(j2[1].tool_calls![1]).toBe(j1[1].tool_calls![1])
  })

  it('works a call out again when its arguments were changed', () => {
    const c = call('a', '{"n":1}')
    const msgs: WireMessage[] = [{ role: 'assistant', content: '', tool_calls: [c] }]
    expect(JSON.parse(jsonifyToolCallArgs(msgs)[0].tool_calls![0].function.arguments)).toEqual({ n: '1' })
    c.function.arguments = '{"n":[1,2]}'
    expect(JSON.parse(jsonifyToolCallArgs(msgs)[0].tool_calls![0].function.arguments)).toEqual({ n: '[1,2]' })
    c.function.arguments = '{"n": "cut'
    expect(validToolCallArgs(msgs)[0].tool_calls![0].function.arguments).toBe('{}')
  })
})

describe('Run my app', () => {
  it('refuses a start command the agent would refuse, without starting anything', async () => {
    const root = mkdtempSync(join(tmpdir(), 'rulz-run-'))
    try {
      const events: RunEvent[] = []
      const runner = new AppRunner(root, (e) => events.push(e), async () => undefined)
      await expect(runner.start('npm start; Stop-Computer -Force')).rejects.toThrow(
        /^I didn't start your app, because its start command \(npm start; Stop-Computer -Force\) isn't safe to run\. It would shut down, restart or sign out of the computer\./
      )
      expect(runner.get()).toMatchObject({ running: false, unknown: true, command: null })
      expect(events.some((e) => e.type === 'state' && e.state.running)).toBe(false)
      // Typed by the owner, it is refused all the same.
      await expect(runner.start('npm start; Stop-Computer -Force', true)).rejects.toThrow(/isn't safe to run/)
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  })

  it('opens only web pages in the project, and refuses the AI a command that hides what it runs', async () => {
    const root = mkdtempSync(join(tmpdir(), 'rulz-run-open-'))
    try {
      const opened: string[] = []
      const runner = new AppRunner(root, () => undefined, async (abs) => void opened.push(abs))
      await expect(runner.start('open:run.bat')).rejects.toThrow(/only opens a web page/)
      await expect(runner.start('open:run.bat', true)).rejects.toThrow(/only opens a web page/)
      expect(opened).toEqual([])
      expect(await runner.start('open:index.html')).toMatchObject({ opened: 'index.html', running: false })
      expect(opened).toEqual([join(root, 'index.html')])
      await expect(runner.start("$c='Write-'+'Output'; return & $c hi")).rejects.toThrow(/isn't safe to run/)
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  })
})
