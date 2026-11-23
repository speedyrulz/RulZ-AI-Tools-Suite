// An unattended practice session in the main process (src/main/re/session.ts), against a scripted model. The
// expectations are what the old loop did: lib/batchRunner.ts driving the chat store (startRe, startGeneration,
// handleEvent, runToolRound, gradePractice, failRePractice), read off that code before it was removed.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { itemResult, REMOTE_DENIAL, sessionStats, UNATTENDED_DENIAL } from '@shared/batch'
import { DEFAULT_SAMPLING } from '@shared/defaults'
import { extractAnswer, RE_SYSTEM_PROMPT, reFirstMessage, reSampling } from '@shared/re'
import { META_LOAD, META_SEARCH, META_UNLOAD } from '@shared/tools'
import type { ChatRequest, Conversation, McpServerStatus, StreamEvent, ToolResult } from '@shared/types'
import { runSession, SAVE_RETRY_WAITS_MS, TRANSPORT_RETRY_MS, type InstanceState, type SessionDeps, type SessionSettings, type SessionSpec } from '../src/main/re/session'

// ---------- fakes ----------

interface Step {
  content?: string
  reasoning?: string
  tools?: { name: string; args?: string }[]
  /** The request fails with this text. */
  error?: string
  transport?: boolean
  /** Never answers; it ends when the request is aborted. */
  hang?: boolean
}

const tool = (name: string, description = `${name} does a thing`) => ({
  qualifiedName: `rea__${name}`,
  server: 'rea',
  name,
  description,
  inputSchema: { type: 'object', properties: {} }
})
const TOOLS = ['open_binary', 'binary_overview', 'search_strings', 'search_procedures', 'procedure_pseudo_code', 'xrefs', 'start_gdb_session'].map((n) => tool(n))
const STARTER = ['rea__open_binary', 'rea__binary_overview', 'rea__search_strings', 'rea__search_procedures', 'rea__procedure_pseudo_code', 'rea__xrefs']
const SERVERS: McpServerStatus[] = [{ name: 'rea', transport: 'stdio', state: 'connected', tools: TOOLS, log: [] }]

const settings = (o: Partial<SessionSettings> = {}): SessionSettings => ({
  toolsEnabled: true,
  toolMode: 'auto',
  toolBudgetTokens: 6000,
  disabledTools: [],
  confirmToolCalls: true,
  allowedTools: [],
  defaultSampling: DEFAULT_SAMPLING,
  contextOverflow: 'truncateMiddle',
  compactToolResults: false,
  keepFullToolResults: 3,
  ...o,
  re: { maxToolRounds: 80, toolBudgetTokens: 16000, kitDir: 'C:\\kit', autoApproveStatic: true, collect: true, ...(o.re ?? {}) }
})

interface Harness {
  deps: SessionDeps
  spec: SessionSpec
  ctrl: AbortController
  requests: ChatRequest[]
  aborted: string[]
  released: string[]
  saves: { conv: Conversation; rev: number }[]
  live: Conversation[]
  toolCalls: { name: string; arguments: string }[]
  clock: { t: number }
  state: { value: InstanceState }
}

function harness(
  script: Step[],
  o: {
    settings?: SessionSettings
    results?: Record<string, ToolResult>
    hangTools?: boolean
    minutes?: number
    state?: () => InstanceState
    watched?: boolean
    grade?: SessionDeps['grade']
    maxToolRounds?: number
  } = {}
): Harness {
  const requests: ChatRequest[] = []
  const aborted: string[] = []
  const released: string[] = []
  const saves: { conv: Conversation; rev: number }[] = []
  const live: Conversation[] = []
  const toolCalls: { name: string; arguments: string }[] = []
  const clock = { t: 1_000_000 }
  const state = { value: 'ready' as InstanceState }
  const hangs = new Map<string, () => void>()
  let step = 0
  let n = 0
  const ctrl = new AbortController()
  const deps: SessionDeps = {
    now: () => (o.minutes ? Date.now() : clock.t),
    sleep: async (ms) => {
      clock.t += ms
    },
    uid: () => `u${++n}`,
    settings: () => o.settings ?? settings(),
    servers: () => SERVERS,
    instance: () => ({ hasVision: false }),
    instanceState: () => (o.state ? o.state() : state.value),
    chat: {
      async start(req, onEvent) {
        requests.push(structuredClone(req))
        const s = script[step++] ?? { content: 'ANSWER: 42' }
        const emit = (e: StreamEvent): void => onEvent(e)
        if (s.hang) {
          await new Promise<void>((resolve) => hangs.set(req.requestId, resolve))
          emit({ type: 'done', requestId: req.requestId, stats: { stopReason: 'stopped', promptTokens: 10, predictedTokens: 1 } })
          return
        }
        if (s.reasoning) emit({ type: 'delta', requestId: req.requestId, reasoning: s.reasoning })
        if (s.content) emit({ type: 'delta', requestId: req.requestId, content: s.content })
        if (s.error) {
          emit({ type: 'error', requestId: req.requestId, error: s.error, stats: { stopReason: 'error' }, ...(s.transport ? { transport: true } : {}) })
          return
        }
        emit({
          type: 'done',
          requestId: req.requestId,
          stats: { stopReason: 'stop', promptTokens: 100 * step, predictedTokens: 10 },
          ...(s.tools
            ? { toolCalls: s.tools.map((t, i) => ({ id: `call_${step}_${i}`, type: 'function' as const, function: { name: t.name, arguments: t.args ?? '{}' } })) }
            : {})
        })
      },
      abort: (id) => {
        aborted.push(id)
        hangs.get(id)?.()
      },
      releaseTurn: (id) => void released.push(id)
    },
    callTool: (call, signal) => {
      toolCalls.push(call)
      if (o.hangTools) {
        return new Promise<ToolResult>((resolve) => signal.addEventListener('abort', () => resolve({ text: 'The tool call was stopped.', isError: true })))
      }
      return Promise.resolve(o.results?.[call.name] ?? { text: `result of ${call.name}`, isError: false })
    },
    grade:
      o.grade ??
      (async (_kit, _id, _target, text) => {
        const answer = extractAnswer(text)
        return { answer, passed: answer === '42', expected: '42' }
      }),
    save: async (conv, rev) => void saves.push({ conv, rev }),
    onLive: (conv) => void live.push(conv),
    watched: () => o.watched ?? false
  }
  const spec: SessionSpec = {
    conversationId: 'chat-1',
    target: 'C:\\work\\xor-1\\1\\xor-1.exe',
    goal: 'Solve C:\\work\\xor-1\\1\\xor-1.exe',
    modelId: 'm/model.gguf',
    instanceId: 'inst-7',
    practice: { taskId: 'xor-1', family: 'xor', tier: 'easy', kitDir: 'C:\\kit' },
    minutesPerTask: o.minutes ?? 0,
    collect: true,
    signal: ctrl.signal,
    ...(o.maxToolRounds ? { maxToolRounds: o.maxToolRounds } : {})
  }
  return { deps, spec, ctrl, requests, aborted, released, saves, live, toolCalls, clock, state }
}

const assistants = (c: Conversation) => c.messages.filter((m) => m.role === 'assistant')

describe('a session on a rented instance', () => {
  const steps: Step[] = [
    { tools: [{ name: 'rea__open_binary', args: '{"path":"C:\\\\Users\\\\u\\\\secret.db"}' }, { name: 'rea__open_binary', args: '{"path":"C:\\\\work\\\\xor-1\\\\1\\\\xor-1.exe"}' }, { name: 'rea__start_gdb_session' }, { name: 'rea__xrefs', args: '{"address":"0x10"}' }] },
    { content: 'ANSWER: 42' }
  ]
  const loose = settings({ confirmToolCalls: false, allowedTools: ['rea__start_gdb_session', 'rea__open_binary'] })

  it('runs static analysis of the task folder only, whatever the approval settings are', async () => {
    for (const how of ['flag', 'model id'] as const) {
      const h = harness(structuredClone(steps), { settings: loose })
      const r = await runSession(h.deps, { ...h.spec, ...(how === 'flag' ? { remote: true } : { modelId: 'vast/m' }) })
      expect(h.toolCalls.map((c) => c.arguments)).toEqual(['{"path":"C:\\\\work\\\\xor-1\\\\1\\\\xor-1.exe"}', '{"address":"0x10"}'])
      const calls = r.conversation.messages.flatMap((m) => m.toolCalls ?? [])
      expect(calls.map((c) => c.status)).toEqual(['denied', 'done', 'denied', 'done'])
      expect(calls[0].result).toBe(REMOTE_DENIAL)
      expect(sessionStats(r.conversation)).toMatchObject({ denied: 2 })
      expect(r.conversation.re?.outcome).toBe('solved')
    }
  })

  it('leaves a model on this PC with the approval settings', async () => {
    const h = harness(structuredClone(steps), { settings: loose })
    await runSession(h.deps, h.spec)
    expect(h.toolCalls).toHaveLength(4)
  })
})

describe('a practice session', () => {
  it('builds the chat the way the chat store did, and solves a task', async () => {
    const h = harness([{ reasoning: 'let me look', tools: [{ name: 'rea__xrefs', args: '{"a":1}' }] }, { content: 'The answer is below.\nANSWER: 42' }])
    const r = await runSession(h.deps, h.spec)
    const c = r.conversation
    // The conversation as startRe made it.
    expect(c).toMatchObject({
      id: 'chat-1',
      title: 'Practice: xor-1',
      folder: null,
      pinned: false,
      modelId: 'm/model.gguf',
      systemPrompt: RE_SYSTEM_PROMPT,
      sampling: reSampling(DEFAULT_SAMPLING),
      adapters: null,
      presetId: null,
      activeTools: STARTER
    })
    expect(c.re).toMatchObject({ target: h.spec.target, goal: h.spec.goal, collect: true, practice: { taskId: 'xor-1', family: 'xor', tier: 'easy', kitDir: 'C:\\kit' } })
    // First request: system prompt and the first message, the model pinned to its instance, recorded.
    const first = h.requests[0]
    expect(first.messages).toEqual([
      { role: 'system', content: RE_SYSTEM_PROMPT },
      { role: 'user', content: reFirstMessage(h.spec.goal, h.spec.target) }
    ])
    expect(first).toMatchObject({ modelId: 'm/model.gguf', instanceId: 'inst-7', adapters: null, overflow: 'truncateMiddle', record: { conversationId: 'chat-1' } })
    expect(first.tools).toEqual({ mode: 'all', active: [] })
    // Second request: the tool call, its result, and the model's thinking kept (interleaved thinking).
    expect(h.requests[1].messages.slice(2)).toEqual([
      {
        role: 'assistant',
        content: '',
        reasoning_content: 'let me look',
        tool_calls: [{ id: 'call_1_0', type: 'function', function: { name: 'rea__xrefs', arguments: '{"a":1}' } }]
      },
      { role: 'tool', tool_call_id: 'call_1_0', content: 'result of rea__xrefs' }
    ])
    // Messages: the user's, the call (with stats), the final reply.
    expect(c.messages.map((m) => m.role)).toEqual(['user', 'assistant', 'assistant'])
    const [callMsg, finalMsg] = assistants(c)
    expect(callMsg.toolCalls).toEqual([{ id: 'call_1_0', name: 'rea__xrefs', arguments: '{"a":1}', status: 'done', result: 'result of rea__xrefs' }])
    expect(callMsg.reasoning).toBe('let me look')
    expect(callMsg.stats).toMatchObject({ stopReason: 'stop', promptTokens: 100, predictedTokens: 10 })
    expect(finalMsg.content).toBe('The answer is below.\nANSWER: 42')
    // Graded, as gradePractice sets it.
    expect(c.re).toMatchObject({ outcome: 'solved', autoGraded: true, practice: { answer: '42', expected: '42' } })
    expect(r).toMatchObject({ end: 'done', timedOut: false, lost: false, aborted: null })
    expect(itemResult(c, r.end, { timedOut: r.timedOut })).toEqual({ status: 'solved', answer: '42' })
    expect(sessionStats(c)).toMatchObject({ toolRounds: 1, promptTokens: 200, outputTokens: 20, denied: 0 })
    // The model was held for the tool round and let go when the next round started.
    expect(h.released).toEqual([h.requests[0].requestId])
  })

  it('saves at the points the store did (a round start, a final reply, the grade), each as the next revision', async () => {
    const h = harness([{ tools: [{ name: 'rea__xrefs' }] }, { content: 'ANSWER: 42' }])
    await runSession(h.deps, h.spec)
    // round 1 start, round 2 start (with the tool results), final reply, grade.
    expect(h.saves.map((s) => s.rev)).toEqual([1, 2, 3, 4])
    expect(h.saves.map((s) => s.conv.poolRev)).toEqual([1, 2, 3, 4])
    // The saved chat of a round start has the user's message and no empty reply.
    expect(h.saves[0].conv.messages.map((m) => m.role)).toEqual(['user'])
    expect(h.saves[1].conv.messages.map((m) => m.role)).toEqual(['user', 'assistant'])
    expect(h.saves[3].conv.re?.outcome).toBe('solved')
  })

  describe('a save that fails', () => {
    it('is tried again after a short wait, and nothing is reported when it then works', async () => {
      const h = harness([{ content: 'ANSWER: 42' }])
      let failures = 2
      const written: number[] = []
      h.deps.save = async (_c, rev) => {
        if (failures-- > 0) throw new Error('EBUSY')
        written.push(rev)
      }
      const r = await runSession(h.deps, h.spec)
      expect(r.saveError).toBeUndefined()
      expect(written.length).toBeGreaterThan(0)
      expect(written.at(-1)).toBe(3)
      // At least the first wait went by (a newer copy may take over from a failed one).
      expect(h.clock.t - 1_000_000).toBeGreaterThanOrEqual(SAVE_RETRY_WAITS_MS[0])
    })

    it('is reported in the result when the final save keeps failing', async () => {
      const h = harness([{ content: 'ANSWER: 42' }])
      const warned: string[] = []
      h.deps.warn = (m) => void warned.push(m)
      h.deps.save = async () => Promise.reject(new Error('disk full'))
      const r = await runSession(h.deps, h.spec)
      expect(r.saveError).toBe('disk full')
      expect(r.end).toBe('done')
      expect(warned.some((m) => m.includes('disk full'))).toBe(true)
    })

    it('does not count an earlier failure when a later save is written', async () => {
      const h = harness([{ tools: [{ name: 'rea__xrefs' }] }, { content: 'ANSWER: 42' }])
      const written: number[] = []
      // Revision 1 never works, the rest do: the later copies hold everything revision 1 had.
      h.deps.save = async (_c, rev) => {
        if (rev === 1) throw new Error('locked')
        written.push(rev)
      }
      const r = await runSession(h.deps, h.spec)
      expect(r.saveError).toBeUndefined()
      expect(written.at(-1)).toBe(4)
    })
  })

  it('grades a wrong answer as wrong and a reply without ANSWER as no answer', async () => {
    const wrong = harness([{ content: 'ANSWER: 41' }])
    const w = await runSession(wrong.deps, wrong.spec)
    expect(w.conversation.re).toMatchObject({ outcome: 'failed', autoGraded: true, practice: { answer: '41', expected: '42' } })
    expect(itemResult(w.conversation, w.end)).toEqual({ status: 'wrong', answer: '41' })
    const none = harness([{ content: 'I could not tell.' }])
    const n = await runSession(none.deps, none.spec)
    expect(n.conversation.re).toMatchObject({ outcome: 'failed', autoGraded: true })
    expect(itemResult(n.conversation, n.end)).toEqual({ status: 'no-answer' })
  })

  it('leaves a session ungraded when grading fails', async () => {
    const h = harness([{ content: 'ANSWER: 42' }], { grade: async () => Promise.reject(new Error('kit gone')) })
    const r = await runSession(h.deps, h.spec)
    expect(r.end).toBe('done')
    expect(itemResult(r.conversation, r.end)).toEqual({ status: 'error', error: 'The final reply could not be graded.' })
  })

  it('stops at the round limit and fails what nobody graded', async () => {
    const h = harness(
      [{ tools: [{ name: 'rea__xrefs' }] }, { tools: [{ name: 'rea__xrefs' }] }, { tools: [{ name: 'rea__xrefs' }] }],
      { settings: settings({ re: { maxToolRounds: 2, toolBudgetTokens: 16000, kitDir: 'C:\\kit', autoApproveStatic: true, collect: true } }) }
    )
    const r = await runSession(h.deps, h.spec)
    expect(r.end).toBe('rounds')
    expect(h.requests).toHaveLength(2)
    expect(r.conversation.re).toMatchObject({ outcome: 'failed', autoGraded: true, notes: 'Batch run: no answer within the tool round limit.' })
    expect(itemResult(r.conversation, r.end)).toEqual({ status: 'rounds' })
  })

  describe('with a time limit', () => {
    beforeEach(() => vi.useFakeTimers())
    afterEach(() => vi.useRealTimers())

    it('stops a session that is still working at the limit, counted from its first request', async () => {
      const h = harness([{ hang: true }], { minutes: 2 })
      const p = runSession(h.deps, h.spec)
      await vi.advanceTimersByTimeAsync(119_000)
      expect(h.aborted).toEqual([])
      await vi.advanceTimersByTimeAsync(2000)
      const r = await p
      expect(h.aborted).toEqual([h.requests[0].requestId])
      expect(r).toMatchObject({ end: 'stopped', timedOut: true, aborted: 'timeout' })
      expect(r.conversation.re).toMatchObject({ outcome: 'failed', autoGraded: true, notes: 'Batch run: stopped at the 2-minute limit.' })
      expect(itemResult(r.conversation, r.end, { timedOut: r.timedOut })).toEqual({ status: 'timeout' })
    })

    it('cancels a tool call that is still running at the limit', async () => {
      const h = harness([{ tools: [{ name: 'rea__xrefs' }] }], { minutes: 1, hangTools: true })
      const p = runSession(h.deps, h.spec)
      await vi.advanceTimersByTimeAsync(61_000)
      const r = await p
      expect(r.timedOut).toBe(true)
      const call = assistants(r.conversation)[0].toolCalls![0]
      expect(call).toMatchObject({ status: 'denied', result: 'Stopped by the user.' })
    })

    it('lets a final reply that beat the limit be graded', async () => {
      // The grade takes longer than the limit: the answer is in, so it is graded and counts.
      let release!: () => void
      const gate = new Promise<void>((r) => (release = r))
      const h = harness([{ content: 'ANSWER: 42' }], {
        minutes: 1,
        grade: async () => {
          await gate
          return { answer: '42', passed: true, expected: '42' }
        }
      })
      const p = runSession(h.deps, h.spec)
      await vi.advanceTimersByTimeAsync(120_000)
      expect(h.aborted).toEqual([])
      release()
      const r = await p
      expect(r).toMatchObject({ end: 'done', timedOut: false })
      expect(itemResult(r.conversation, r.end, { timedOut: r.timedOut })).toEqual({ status: 'solved', answer: '42' })
    })

    it('has no limit at 0 minutes', async () => {
      const h = harness([{ content: 'ANSWER: 42' }], { minutes: 0 })
      const r = await runSession(h.deps, h.spec)
      expect(r.timedOut).toBe(false)
    })
  })

  it('records a failed request on the reply and ends as an error', async () => {
    const h = harness([{ content: 'partial', error: 'The model fell over.' }])
    const r = await runSession(h.deps, h.spec)
    expect(r.end).toBe('error')
    const [m] = assistants(r.conversation)
    expect(m).toMatchObject({ content: 'partial', error: 'The model fell over.' })
    expect(itemResult(r.conversation, r.end)).toEqual({ status: 'error', error: 'The model fell over.' })
  })

  it('refuses a tool that needs approval, tells the model, and goes on', async () => {
    const h = harness([{ tools: [{ name: 'rea__start_gdb_session' }, { name: 'xrefs' }] }, { content: 'ANSWER: 42' }])
    const r = await runSession(h.deps, h.spec)
    const calls = assistants(r.conversation)[0].toolCalls!
    expect(calls[0]).toMatchObject({ name: 'rea__start_gdb_session', status: 'denied', result: UNATTENDED_DENIAL })
    // A bare name is mapped to the real tool, and renamed on the message.
    expect(calls[1]).toMatchObject({ name: 'rea__xrefs', status: 'done' })
    expect(h.toolCalls).toEqual([{ name: 'rea__xrefs', arguments: '{}' }])
    expect(sessionStats(r.conversation).denied).toBe(1)
    expect(h.requests[1].messages.filter((m) => m.role === 'tool')[0].content).toBe(UNATTENDED_DENIAL)
  })

  it('runs a tool the user allowed, and reports unknown tools and tool errors', async () => {
    const h = harness([{ tools: [{ name: 'rea__start_gdb_session' }, { name: 'rea__nope' }, { name: 'rea__xrefs' }] }, { content: 'ANSWER: 42' }], {
      settings: settings({ allowedTools: ['rea__start_gdb_session'] }),
      results: { rea__xrefs: { text: 'boom', isError: true } }
    })
    const r = await runSession(h.deps, h.spec)
    const calls = assistants(r.conversation)[0].toolCalls!
    expect(calls[0].status).toBe('done')
    expect(calls[1]).toMatchObject({ status: 'error', result: 'Tool "rea__nope" is not available or is switched off.' })
    expect(calls[2]).toMatchObject({ status: 'error', result: 'boom' })
  })

  describe('on-demand tools', () => {
    const dynamic = (budget: number) => settings({ toolMode: 'dynamic', re: { maxToolRounds: 80, toolBudgetTokens: budget, kitDir: 'C:\\kit', autoApproveStatic: true, collect: true } })

    it('sends the loaded tools, searches, loads with least-recently-used eviction, and unloads', async () => {
      const total = (names: string[]) =>
        names.reduce((n, q) => n + Math.ceil(JSON.stringify({ name: q, description: `${q.slice(5)} does a thing`, parameters: { type: 'object', properties: {} } }).length / 3.5), 0)
      const all = [...STARTER, 'rea__start_gdb_session']
      // One short of everything: loading the 7th tool pushes out exactly the oldest.
      const h = harness(
        [
          { tools: [{ name: META_SEARCH, args: '{"query":"xrefs"}' }] },
          { tools: [{ name: META_LOAD, args: '{"names":["start_gdb_session","nonexistent"]}' }] },
          { tools: [{ name: META_UNLOAD, args: '{"names":["rea__xrefs"]}' }] },
          { content: 'ANSWER: 42' }
        ],
        { settings: dynamic(total(all) - 1) }
      )
      const r = await runSession(h.deps, h.spec)
      expect(h.requests[0].tools).toEqual({ mode: 'dynamic', active: STARTER })
      const results = assistants(r.conversation).map((m) => m.toolCalls?.[0]?.result)
      expect(results[0]).toBe('- xrefs (rea) [loaded]: xrefs does a thing')
      expect(results[1]).toBe(
        'Loaded: rea__start_gdb_session. You can call them in your next step.\n' +
          'Unloaded to stay within the tool budget (least recently used): rea__open_binary.\n' +
          'Unknown tools: nonexistent.'
      )
      expect(results[2]).toBe('Unloaded: rea__xrefs.')
      expect(h.requests[2].tools).toEqual({ mode: 'dynamic', active: STARTER.filter((n) => n !== 'rea__open_binary').concat('rea__start_gdb_session') })
      expect(r.conversation.activeTools).toEqual(['rea__binary_overview', 'rea__search_strings', 'rea__search_procedures', 'rea__procedure_pseudo_code', 'rea__start_gdb_session'])
      // The loader tools are not sent to the MCP servers.
      expect(h.toolCalls).toEqual([])
    })

    it('keeps a tool loaded that the model calls', async () => {
      const h = harness([{ tools: [{ name: 'rea__start_gdb_session' }] }, { content: 'ANSWER: 42' }], { settings: dynamic(100000) })
      const r = await runSession(h.deps, h.spec)
      // Denied (it asks), but calling it marks it loaded and most recent first.
      expect(r.conversation.activeTools?.at(-1)).toBe('rea__start_gdb_session')
    })

    it('reports bad loader arguments', async () => {
      const h = harness([{ tools: [{ name: META_LOAD, args: 'not json' }, { name: META_LOAD, args: '{"names":[]}' }] }, { content: 'ANSWER: 42' }], { settings: dynamic(100000) })
      const r = await runSession(h.deps, h.spec)
      const [a, b] = assistants(r.conversation)[0].toolCalls!
      expect(a).toMatchObject({ status: 'error', result: 'The arguments were not valid JSON: not json' })
      expect(b.status).toBe('error')
      expect(b.result).toMatch(/^No tool names arrived/)
    })
  })

  it('shortens old tool results as the settings say, keeping the newest whole', async () => {
    const big = (c: string): ToolResult => ({ text: c.repeat(2000), isError: false })
    const h = harness([{ tools: [{ name: 'rea__xrefs' }] }, { tools: [{ name: 'rea__search_strings' }] }, { content: 'ANSWER: 42' }], {
      settings: settings({ compactToolResults: true, keepFullToolResults: 1 }),
      results: { rea__xrefs: big('x'), rea__search_strings: big('y') }
    })
    const r = await runSession(h.deps, h.spec)
    const toolMsgs = h.requests[2].messages.filter((m) => m.role === 'tool')
    expect(toolMsgs).toHaveLength(2)
    expect(String(toolMsgs[0].content)).toContain('older tool result omitted to save context')
    expect(String(toolMsgs[0].content).length).toBeLessThan(1000)
    expect(toolMsgs[1].content).toBe('y'.repeat(2000))
    // The chat itself keeps the whole result.
    expect(assistants(r.conversation)[0].toolCalls![0].result).toHaveLength(2000)
  })

  describe('when the instance cannot be reached', () => {
    it('drops the partial reply, waits while it reconnects, and sends the round again', async () => {
      const seen: InstanceState[] = ['reconnecting', 'reconnecting', 'ready']
      const h = harness([{ content: 'half a reply', error: 'fetch failed', transport: true }, { content: 'ANSWER: 42' }], { state: () => seen.shift() ?? 'ready' })
      const r = await runSession(h.deps, h.spec)
      expect(r).toMatchObject({ end: 'done', lost: false })
      expect(h.requests).toHaveLength(2)
      // The same history, and no trace of the dropped reply.
      expect(h.requests[1].messages).toEqual(h.requests[0].messages)
      expect(assistants(r.conversation)).toHaveLength(1)
      expect(assistants(r.conversation)[0].content).toBe('ANSWER: 42')
      expect(h.requests.every((q) => q.instanceId === 'inst-7')).toBe(true)
    })

    it('gives up after 3 minutes and reports the session lost', async () => {
      const h = harness([{ error: 'fetch failed', transport: true }], { state: () => 'reconnecting' })
      const r = await runSession(h.deps, h.spec)
      expect(r).toMatchObject({ lost: true, end: 'error' })
      expect(h.clock.t - 1_000_000).toBeGreaterThanOrEqual(TRANSPORT_RETRY_MS)
      expect(h.requests).toHaveLength(1)
      expect(assistants(r.conversation)).toHaveLength(0)
    })

    it('is lost at once when the instance is gone', async () => {
      const h = harness([{ error: 'fetch failed', transport: true }], { state: () => 'gone' })
      const r = await runSession(h.deps, h.spec)
      expect(r.lost).toBe(true)
      expect(h.clock.t).toBe(1_000_000)
    })

    it('does not count the wait against the time limit', async () => {
      vi.useFakeTimers()
      try {
        let waited = 0
        // Reconnecting for 90 s: longer than the 1-minute limit.
        const h = harness([{ error: 'fetch failed', transport: true }, { content: 'ANSWER: 42' }], { minutes: 1, state: () => (waited >= 90_000 ? 'ready' : 'reconnecting') })
        h.deps.sleep = async (ms) => {
          await vi.advanceTimersByTimeAsync(ms)
          waited += ms
        }
        const r = await runSession(h.deps, h.spec)
        expect(r).toMatchObject({ end: 'done', timedOut: false, lost: false })
      } finally {
        vi.useRealTimers()
      }
    })
  })

  describe('stopping', () => {
    it('"skip" ends a reply being written, keeps what it had, and reports it stopped', async () => {
      const h = harness([{ hang: true }])
      const p = runSession(h.deps, h.spec)
      await Promise.resolve()
      h.ctrl.abort('skip')
      const r = await p
      expect(r).toMatchObject({ end: 'stopped', aborted: 'skip', timedOut: false })
      expect(itemResult(r.conversation, r.end)).toEqual({ status: 'stopped' })
      // A skipped task is not failed by the batch notes.
      expect(r.conversation.re?.outcome).toBe('unrated')
    })

    it('"stopNow" cancels the running tool call and marks the rest of the round stopped', async () => {
      const h = harness([{ tools: [{ name: 'rea__xrefs' }, { name: 'rea__binary_overview' }] }], { hangTools: true })
      const p = runSession(h.deps, h.spec)
      await new Promise((r) => setTimeout(r, 5))
      h.ctrl.abort('stopNow')
      const r = await p
      expect(r).toMatchObject({ end: 'stopped', aborted: 'stopNow' })
      const calls = assistants(r.conversation)[0].toolCalls!
      expect(calls.map((c) => c.status)).toEqual(['denied', 'denied'])
      expect(calls.map((c) => c.result)).toEqual(['Stopped by the user.', 'Stopped by the user.'])
      expect(h.toolCalls).toHaveLength(1)
    })

    it('does not start when it is already aborted', async () => {
      const h = harness([{ content: 'ANSWER: 42' }])
      h.ctrl.abort('stopNow')
      const r = await runSession(h.deps, h.spec)
      expect(r).toMatchObject({ end: 'stopped', aborted: 'stopNow' })
      expect(h.requests).toHaveLength(0)
    })
  })

  describe('live updates', () => {
    it('sends nothing when nobody watches', async () => {
      const h = harness([{ content: 'ANSWER: 42' }], { watched: false })
      await runSession(h.deps, h.spec)
      expect(h.live).toHaveLength(0)
    })

    it('sends the chat when it is watched, the streamed text included, at most 4 times a second', async () => {
      const h = harness([{ content: 'ANSWER: 42' }], { watched: true })
      // Many deltas in no time at all: still only a few updates.
      const start = h.deps.chat.start
      h.deps.chat.start = async (req, onEvent) => {
        queueMicrotask(() => {
          for (let i = 0; i < 200; i++) onEvent({ type: 'delta', requestId: req.requestId, content: 'x' })
        })
        return start(req, onEvent)
      }
      await runSession(h.deps, h.spec)
      expect(h.live.length).toBeGreaterThan(0)
      expect(h.live.length).toBeLessThan(10)
      expect(assistants(h.live.at(-1)!)[0].content).toContain('ANSWER: 42')
    })
  })
})
