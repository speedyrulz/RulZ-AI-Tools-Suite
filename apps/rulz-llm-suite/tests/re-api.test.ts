import http from 'node:http'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import { API_RE_PREFIX, RE_SESSION_HEADER } from '@shared/re'
import type { LoadedInstance, ModelEntry, ReOverview, ReSession } from '@shared/types'
import type { RecordedRound } from '../src/main/chat/service'
import type { InstanceManager } from '../src/main/engine/instances'
import { exportDataset, TraceStore } from '../src/main/re/traces'
import { buildOverview } from '../src/main/re/overview'
import { ApiServer, parseReSessionPatch } from '../src/main/server/apiServer'
import { ChatReplyCapture } from '../src/main/server/reCapture'

const freePort = async (): Promise<number> => {
  const s = http.createServer()
  await new Promise<void>((r) => s.listen(0, '127.0.0.1', () => r()))
  const port = (s.address() as { port: number }).port
  await new Promise<void>((r) => s.close(() => r()))
  return port
}
const sse = (...events: unknown[]): Buffer => Buffer.from(events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('') + 'data: [DONE]\n\n')
const delta = (d: Record<string, unknown>, finish: string | null = null) => ({ choices: [{ index: 0, delta: d, finish_reason: finish }] })

describe('ChatReplyCapture', () => {
  it('collects content, thinking and tool calls from a stream split anywhere', () => {
    const body = sse(
      delta({ reasoning_content: 'look ' }),
      delta({ reasoning_content: 'at main' }),
      delta({ tool_calls: [{ index: 0, id: 'c1', type: 'function', function: { name: 'rea__open', arguments: '{"pa' } }] }),
      delta({ tool_calls: [{ index: 0, function: { arguments: 'th":"a.exe"}' } }] }),
      delta({ tool_calls: [{ index: 1, id: 'c2', function: { name: 'calc__calc', arguments: '{}' } }] }),
      delta({}, 'tool_calls')
    )
    const cap = new ChatReplyCapture(true)
    // Byte-sized pieces, so events and multi-byte characters are cut mid-way.
    for (let i = 0; i < body.length; i += 7) cap.push(body.subarray(i, i + 7))
    expect(cap.result()).toEqual({
      content: '',
      reasoning: 'look at main',
      toolCalls: [
        { id: 'c1', type: 'function', function: { name: 'rea__open', arguments: '{"path":"a.exe"}' } },
        { id: 'c2', type: 'function', function: { name: 'calc__calc', arguments: '{}' } }
      ],
      finishReason: 'tool_calls'
    })
  })

  it('reads a non-streamed completion and ignores what is not one', () => {
    const cap = new ChatReplyCapture(false)
    cap.push(Buffer.from(JSON.stringify({ choices: [{ message: { content: 'ANSWER: 42', reasoning_content: 'hm' }, finish_reason: 'stop' }] })))
    expect(cap.result()).toEqual({ content: 'ANSWER: 42', reasoning: 'hm', toolCalls: [], finishReason: 'stop' })
    const junk = new ChatReplyCapture(false)
    junk.push(Buffer.from('{"error":"x"}'))
    expect(junk.result()).toBeNull()
  })
})

describe('RE session descriptions', () => {
  it('accepts the known fields and outcomes only', () => {
    expect(parseReSessionPatch({ target: 'C:\\x.exe', goal: 'g', outcome: 'solved' })).toEqual({ target: 'C:\\x.exe', goal: 'g', outcome: 'solved' })
    expect(typeof parseReSessionPatch({ outcome: 'great' })).toBe('string')
    expect(typeof parseReSessionPatch({ title: 5 })).toBe('string')
    expect(parseReSessionPatch({ other: 'ignored' })).toEqual({})
  })
})

describe('recording RE sessions from the API server', () => {
  let dir: string
  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), 'rulz-re-api-'))
  })
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true })
  })

  it('records marked chat completions, describes and grades the session, and exports it', async () => {
    const replies = [
      sse(delta({ reasoning_content: 'open it' }), delta({ tool_calls: [{ index: 0, id: 'c1', function: { name: 'rea__open_binary', arguments: '{}' } }] }), delta({}, 'tool_calls')),
      sse(delta({ reasoning_content: 'done' }), delta({ content: 'ANSWER: 7' }), delta({}, 'stop'))
    ]
    let n = 0
    const upstream = http.createServer((req, res) => {
      req.resume()
      req.on('end', () => res.writeHead(200, { 'Content-Type': 'text/event-stream' }).end(replies[Math.min(n++, replies.length - 1)]))
    })
    await new Promise<void>((r) => upstream.listen(0, '127.0.0.1', () => r()))
    const upstreamPort = (upstream.address() as { port: number }).port
    const model: ModelEntry = { id: 'M/m.gguf', name: 'm', folder: 'M', path: 'C:\\LLM\\M\\m.gguf', files: [], sizeBytes: 1, info: null, modifiedAt: 0 }
    const inst = { instanceId: 'i1', modelId: model.id, status: 'ready', port: upstreamPort, config: DEFAULT_LOAD_CONFIG, contextLength: 8192 } as unknown as LoadedInstance
    const instances = { list: () => [inst], readyFor: () => inst, acquire: () => () => undefined, idleSince: () => null } as unknown as InstanceManager

    const traces = new TraceStore(join(dir, 're-data'))
    let enabled = true
    const recorded: Promise<void>[] = []
    const port = await freePort()
    const server = new ApiServer({
      instances,
      getModels: () => [model],
      getSettings: () => ({ autoStart: false, port, bind: 'localhost', apiKey: '', cors: false, idleUnloadMinutes: 0, defaultModelId: '', unknownModelFallback: true }),
      getLoadDefaults: () => ({ config: DEFAULT_LOAD_CONFIG, custom: false }),
      setLoadDefaults: async () => undefined,
      estimate: () => null,
      gpus: () => [],
      onStatus: () => undefined,
      onRequest: () => undefined,
      re: {
        enabled: () => enabled,
        record: (id: string, round: RecordedRound) => void recorded.push(traces.record(API_RE_PREFIX + id, round, { shrinkIsTrim: true })),
        session: (id: string) => traces.apiSession(API_RE_PREFIX + id),
        update: (id: string, patch) => traces.updateApiSession(API_RE_PREFIX + id, patch),
        overview: async () => buildOverview({ chats: [], traces: await traces.list(), practice: { kitDir: 'k', tasks: [] }, recording: enabled }),
        preparePractice: async (taskId: string) => {
          if (taskId !== 'xor-1') throw new Error('No such task.')
          return { target: 'C:\\w\\xor.exe', prompt: 'Find the key in C:\\w\\xor.exe', family: 'xor', tier: 't1' }
        },
        gradePractice: async (_taskId: string, _target: string, text: string) => ({ answer: text.includes('ANSWER: 7') ? '7' : null, passed: text.includes('ANSWER: 7'), expected: '7' })
      }
    })
    await server.start()
    const base = `http://127.0.0.1:${port}`
    try {
      const describe = await fetch(`${base}/v1/re/sessions/task-1`, {
        method: 'POST',
        body: JSON.stringify({ client: 'RulZ Coding Agent', title: 'Find the key', target: 'C:\\t\\a.exe', goal: 'What is the key?' })
      })
      expect(describe.status).toBe(200)
      expect(await describe.json()).toMatchObject({ id: 'task-1', recording: true, client: 'RulZ Coding Agent', outcome: 'unrated' })
      expect((await fetch(`${base}/v1/re/sessions/..%2Fx`)).status).toBe(400)

      const tools = [{ type: 'function', function: { name: 'rea__open_binary', parameters: { type: 'object' } } }]
      const user = { role: 'user', content: 'What is the key?' }
      const chat = (messages: unknown[], headers: Record<string, string> = { [RE_SESSION_HEADER]: 'task-1' }) =>
        fetch(`${base}/v1/chat/completions`, { method: 'POST', headers: { 'Content-Type': 'application/json', ...headers }, body: JSON.stringify({ model: 'm', stream: true, messages, tools }) }).then((r) => r.text())
      await chat([user])
      const call = { role: 'assistant', content: '', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'rea__open_binary', arguments: '{}' } }] }
      await chat([user, call, { role: 'tool', tool_call_id: 'c1', content: 'ok' }])
      // Unmarked requests and requests while recording is off are not recorded.
      await chat([user], {})
      enabled = false
      await chat([user])
      await Promise.all(recorded)
      expect(recorded).toHaveLength(2)

      const info = await traces.info(API_RE_PREFIX + 'task-1')
      expect(info).toMatchObject({ responses: 2, toolRounds: 1, trimmed: false, api: { client: 'RulZ Coding Agent', target: 'C:\\t\\a.exe' } })

      // Grading works while recording is off; plain updates don't create sessions then.
      expect((await (await fetch(`${base}/v1/re/sessions/task-1`, { method: 'POST', body: '{"outcome":"solved"}' })).json()) as object).toMatchObject({ outcome: 'solved', recording: false })
      expect((await fetch(`${base}/v1/re/sessions/task-2`, { method: 'POST', body: '{"title":"x"}' })).status).toBe(200)
      expect(await traces.apiSession(API_RE_PREFIX + 'task-2')).toBeNull()

      const overview = (await (await fetch(`${base}/v1/re/overview`)).json()) as ReOverview
      expect(overview.sessions).toMatchObject([{ id: 'api-task-1', source: 'RulZ Coding Agent', kind: 'analyse', outcome: 'solved', toolRounds: 1 }])
      const prep = await fetch(`${base}/v1/re/practice/xor-1/prepare`, { method: 'POST' })
      expect(await prep.json()).toMatchObject({ target: 'C:\\w\\xor.exe', family: 'xor' })
      expect((await fetch(`${base}/v1/re/practice/nope/prepare`, { method: 'POST' })).status).toBe(400)
      expect((await fetch(`${base}/v1/re/practice/xor-1/run`, { method: 'POST' })).status).toBe(404)
      const grade = await fetch(`${base}/v1/re/practice/xor-1/grade`, { method: 'POST', body: JSON.stringify({ target: 'C:\\w\\xor.exe', text: 'so\nANSWER: 7' }) })
      expect(await grade.json()).toEqual({ answer: '7', passed: true, expected: '7' })
      expect((await fetch(`${base}/v1/re/practice/xor-1/grade`, { method: 'POST', body: '{}' })).status).toBe(400)

      const api = (await traces.apiSession(API_RE_PREFIX + 'task-1'))!
      const r = await exportDataset(traces, [{ id: API_RE_PREFIX + 'task-1', source: api.client, re: { target: api.target, goal: api.goal, collect: true, outcome: api.outcome } }], {
        outDir: join(dir, 'out'),
        outcomes: ['solved'],
        skipTrimmed: true,
        maxPerTask: 2
      })
      expect(r.kept).toBe(1)
      const sample = JSON.parse((await readFile(join(dir, 'out', 'train.jsonl'), 'utf8')).trim())
      expect(sample.messages.map((m: { role: string }) => m.role)).toEqual(['user', 'assistant', 'tool', 'assistant'])
      expect(sample.messages[1].reasoning_content).toBe('open it')
      expect(sample.messages[3]).toEqual({ role: 'assistant', content: 'ANSWER: 7', reasoning_content: 'done' })
      expect(sample.tools).toEqual(tools)
      expect(sample.meta).toMatchObject({ conversation: 'api-task-1', source: 'RulZ Coding Agent', outcome: 'solved', model: 'm' })
    } finally {
      await server.stop()
      await new Promise<void>((r) => upstream.close(() => r()))
    }
    // Two servers, real HTTP and file writes: it passed in about 3 s alone but timed out at the default 5 s on a busy machine.
  }, 20_000)

  it('counts a request with fewer messages than the last one as trimmed', async () => {
    const traces = new TraceStore(join(dir, 're-data'))
    const round = (messages: number): RecordedRound => ({
      requestId: String(messages),
      at: messages,
      model: 'm',
      messages: Array.from({ length: messages }, () => ({ role: 'user' as const, content: 'x' })),
      tools: [],
      trimmed: false,
      response: { content: 'a', reasoning: '', toolCalls: [] }
    })
    await traces.record('api-a', round(3), { shrinkIsTrim: true })
    await traces.record('api-a', round(5), { shrinkIsTrim: true })
    expect((await traces.info('api-a'))?.trimmed).toBe(false)
    await traces.record('api-a', round(2), { shrinkIsTrim: true })
    expect((await traces.info('api-a'))?.trimmed).toBe(true)
    // In-app chats don't use the rule (an edited message makes a shorter request).
    await traces.record('chat', round(5))
    await traces.record('chat', round(2))
    expect((await traces.info('chat'))?.trimmed).toBe(false)
  })
})

describe('RE overview', () => {
  const re = (extra: Partial<ReSession> = {}): ReSession => ({ target: 'C:\\a.exe', goal: 'g', collect: true, outcome: 'unrated', ...extra })
  it('lists chats and API sessions newest first, with practice counts', () => {
    const o = buildOverview({
      chats: [
        { id: 'c1', title: 'RE: a', updatedAt: 10, re: re({ outcome: 'solved', practice: { taskId: 'xor-1', family: 'xor', tier: 't1', answer: '7', expected: '7' } }) },
        { id: 'c2', title: 'Clone: a', updatedAt: 30, re: re({ clone: { scope: 'full', details: '', language: 'C', outputDir: 'D:\\o' } }) }
      ],
      traces: [
        { conversationId: 'c1', responses: 4, toolRounds: 3, trimmed: false, bytes: 1, updatedAt: 12 },
        {
          conversationId: 'api-agent-1',
          responses: 2,
          toolRounds: 1,
          trimmed: false,
          bytes: 1,
          updatedAt: 20,
          api: { client: 'RulZ Coding Agent', title: 'x', target: 't', goal: 'g', outcome: 'failed', createdAt: 1, practice: { taskId: 'xor-1', family: 'xor', tier: 't1' } }
        }
      ],
      practice: { kitDir: 'k', tasks: [{ id: 'xor-1', family: 'xor', tier: 't1', binary: 'xor.exe', prompt: 'p' }, { id: 'xor-2', family: 'xor', tier: 't2', binary: 'b.exe', prompt: 'q' }] },
      recording: true
    })
    expect(o.sessions.map((s) => [s.id, s.kind, s.source])).toEqual([
      ['c2', 'clone-full', 'RulZ LLM Suite'],
      ['api-agent-1', 'practice', 'RulZ Coding Agent'],
      ['c1', 'practice', 'RulZ LLM Suite']
    ])
    expect(o.sessions[2]).toMatchObject({ responses: 4, toolRounds: 3, updatedAt: 12 })
    expect(o.practice.tasks.map((t) => [t.id, t.attempts, t.solved])).toEqual([
      ['xor-1', 2, 1],
      ['xor-2', 0, 0]
    ])
    expect(o.goals.length).toBeGreaterThan(0)
  })
  it('accepts kind and practice in a session description', () => {
    expect(parseReSessionPatch({ kind: 'practice', practice: { taskId: 'a', family: 'f', tier: 't', answer: '1' } })).toEqual({
      kind: 'practice',
      practice: { taskId: 'a', family: 'f', tier: 't', answer: '1' }
    })
    expect(parseReSessionPatch({ kind: 'other' })).toMatch(/kind/)
    expect(parseReSessionPatch({ practice: { taskId: 1 } })).toMatch(/practice/)
  })
})
