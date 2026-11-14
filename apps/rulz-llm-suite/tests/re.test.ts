import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { extractAnswer, isStaticReTool, reFirstMessage, reSampling } from '@shared/re'
import { DEFAULT_SAMPLING } from '@shared/defaults'
import type { ChatMessage, Conversation, ReSession, WireMessage } from '@shared/types'
import type { RecordedRound } from '../src/main/chat/service'
import { buildSample, exportDataset, sampleLine, scrub, TraceStore, type RequestFile, type ResponseLine } from '../src/main/re/traces'
import { checkAnswer, gradePractice, listPractice, preparePractice, type KitTask } from '../src/main/re/practice'
import { parseSettings } from '../src/main/settings/schema'
import { buildWire } from '../src/renderer/src/lib/wire'

const call = (id: string, name: string, args = '{}') => ({ id, type: 'function' as const, function: { name, arguments: args } })
const tool = (name: string) => ({ type: 'function', function: { name, description: name, parameters: { type: 'object' } } })

describe('RE answers and tools', () => {
  it('reads the last ANSWER line like the kit does', () => {
    expect(extractAnswer('foo\nANSWER: 1234\nbar\n**ANSWER**: `8246C3B6`')).toBe('8246C3B6')
    expect(extractAnswer('answer: "zephyr-ember-18"')).toBe('zephyr-ember-18')
    expect(extractAnswer('no answer here')).toBeNull()
    expect(extractAnswer('ANSWER:   ')).toBeNull()
  })

  it('auto-approves only static analysis tools', () => {
    expect(isStaticReTool('rea', 'procedure_pseudo_code')).toBe(true)
    expect(isStaticReTool('calc', 'calc')).toBe(true)
    expect(isStaticReTool('rulz', 'list_directory')).toBe(true)
    for (const n of ['execute_rizin_command', 'cutter_command', 'start_frida_session', 'install_adb_package', 'capture_process_scenario', 'start_gdb_session']) {
      expect(isStaticReTool('rea', n)).toBe(false)
    }
    expect(isStaticReTool('other', 'xrefs')).toBe(false)
  })

  it('puts the target in the first message', () => {
    expect(reFirstMessage('Analyze {target}.', 'C:\\a.exe')).toBe('Analyze C:\\a.exe.')
    expect(reFirstMessage('What does it do?', 'C:\\a.exe')).toBe('Target: C:\\a.exe\n\nWhat does it do?')
  })

  it('uses thinking sampling without a repeat penalty', () => {
    const s = reSampling(DEFAULT_SAMPLING)
    expect(s).toMatchObject({ temperature: 0.6, topK: 20, topP: 0.95, minP: 0, repeatPenalty: 1, thinking: 'on' })
  })

  it('fills RE settings in for older settings files', () => {
    const s = parseSettings({ modelsDir: 'D:\\m' })
    expect(s.re.maxToolRounds).toBe(80)
    expect(parseSettings({ re: { maxToolRounds: -3, collect: false } }).re).toMatchObject({ maxToolRounds: 80, collect: false })
  })
})

describe('sending thinking back in RE sessions', () => {
  const msgs: ChatMessage[] = [
    { id: 'u1', role: 'user', content: 'old', createdAt: 0 },
    { id: 'a1', role: 'assistant', content: 'old answer', reasoning: 'old thought', createdAt: 0 },
    { id: 'u2', role: 'user', content: 'analyze', createdAt: 0 },
    {
      id: 'a2',
      role: 'assistant',
      content: '',
      reasoning: 'open it first',
      toolCalls: [{ id: 'c1', name: 'rea__open_binary', arguments: '{}', status: 'done', result: 'ok' }],
      createdAt: 0
    }
  ]
  it('keeps the thinking of the current tool loop only', () => {
    const wire = buildWire(msgs, 'sys', false, null, true)
    expect(wire.find((m) => m.content === 'old answer')?.reasoning_content).toBeUndefined()
    expect(wire.find((m) => m.tool_calls)?.reasoning_content).toBe('open it first')
  })
  it('sends no thinking in ordinary chats', () => {
    expect(buildWire(msgs, 'sys', false, null).some((m) => m.reasoning_content)).toBe(false)
  })
})

describe('building training samples', () => {
  const messages: WireMessage[] = [
    { role: 'system', content: 'sys' },
    { role: 'user', content: 'find the key' },
    { role: 'assistant', content: '', tool_calls: [call('c1', 'rea__open_binary')] },
    { role: 'tool', tool_call_id: 'c1', content: 'opened' },
    { role: 'assistant', content: '', reasoning_content: 'sent back', tool_calls: [call('c2', 'calc__calc')] },
    { role: 'tool', tool_call_id: 'c2', content: '42' }
  ]
  const request: RequestFile = { requestId: 'r3', at: 3, model: 'ornith', messages, tools: [tool('calc__calc')], trimmedEver: false }
  const responses: ResponseLine[] = [
    { requestId: 'r1', at: 1, content: '', reasoning: 'open it', toolCalls: [call('c1', 'rea__open_binary')] },
    { requestId: 'r2', at: 2, content: '', reasoning: 'compute', toolCalls: [call('c2', 'calc__calc')] },
    { requestId: 'r3', at: 3, content: 'The key is 42.\nANSWER: 42', reasoning: 'done', toolCalls: [] }
  ]

  it('adds the final reply and the thinking the app did not send back', () => {
    const s = buildSample(request, responses, { rea__open_binary: tool('rea__open_binary') }, { outcome: 'solved' })
    if (typeof s === 'string') throw new Error(s)
    expect(s.messages[2].reasoning_content).toBe('open it')
    expect(s.messages[4].reasoning_content).toBe('sent back')
    expect(s.messages.at(-1)).toEqual({ role: 'assistant', content: 'The key is 42.\nANSWER: 42', reasoning_content: 'done' })
    // A tool unloaded before the end is still described, since the session called it.
    expect(s.tools.map((t) => (t as { function: { name: string } }).function.name)).toEqual(['calc__calc', 'rea__open_binary'])
    expect(s.meta).toMatchObject({ outcome: 'solved', model: 'ornith' })
  })

  it('refuses sessions that did not end in an answer', () => {
    expect(buildSample(request, responses.slice(0, 2), {}, {})).toMatch(/no recorded reply/)
    const onTool = [...responses.slice(0, 2), { ...responses[2], toolCalls: [call('c3', 'calc__calc')] }]
    expect(buildSample(request, onTool, {}, {})).toMatch(/tool call/)
    expect(buildSample(request, [...responses.slice(0, 2), { ...responses[2], finishReason: 'length' }], {}, {})).toMatch(/cut off/)
  })

  it('replaces email addresses', () => {
    expect(scrub('{"a":"me@example.org wrote"}')).toBe('{"a":"user@example.com wrote"}')
  })

  it('keeps the line valid JSON when an address follows a newline or tab', () => {
    const sample = { messages: [{ role: 'tool' as const, tool_call_id: 'c1', content: 'strings:\nsupport@vendor.com\ttech@vendor.com' }], tools: [], meta: {} }
    const parsed = JSON.parse(sampleLine(sample))
    expect(parsed.messages[0].content).toBe('strings:\nuser@example.com\tuser@example.com')
  })
})

describe('recording and exporting', () => {
  let dir: string
  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), 'rulz-re-test-'))
  })
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true })
  })

  const round = (requestId: string, messages: WireMessage[], response: Partial<RecordedRound['response']>, trimmed = false): RecordedRound => ({
    requestId,
    at: Date.now(),
    model: 'ornith',
    messages,
    tools: [tool('rea__open_binary')],
    trimmed,
    response: { content: '', reasoning: '', toolCalls: [], ...response }
  })

  const conv = (id: string, re: Partial<ReSession>): Conversation => ({
    id,
    title: id,
    folder: null,
    pinned: false,
    createdAt: 0,
    updatedAt: 0,
    modelId: null,
    systemPrompt: '',
    sampling: DEFAULT_SAMPLING,
    presetId: null,
    messages: [],
    re: { target: 'x.exe', goal: 'g', collect: true, outcome: 'solved', ...re }
  })

  it('records rounds and exports graded sessions as train.jsonl', async () => {
    const store = new TraceStore(join(dir, 'data'))
    const base: WireMessage[] = [{ role: 'user', content: 'go' }]
    const asked: WireMessage[] = [...base, { role: 'assistant', content: '', tool_calls: [call('c1', 'rea__open_binary')] }, { role: 'tool', tool_call_id: 'c1', content: 'ok' }]
    for (const id of ['good', 'bad', 'cut']) {
      await store.record(id, round('r1', base, { reasoning: 'think', toolCalls: [call('c1', 'rea__open_binary')] }, id === 'cut'))
      await store.record(id, round('r2', asked, { content: 'ANSWER: 1', reasoning: 'final' }))
    }
    const info = await store.info('good')
    expect(info).toMatchObject({ responses: 2, toolRounds: 1, trimmed: false })
    expect((await store.info('cut'))?.trimmed).toBe(true)

    const out = join(dir, 'out')
    const r = await exportDataset(store, [conv('good', {}), conv('bad', { outcome: 'failed' }), conv('cut', {}), conv('none', {})], {
      outDir: out,
      outcomes: ['solved'],
      skipTrimmed: true,
      maxPerTask: 2
    })
    expect(r.kept).toBe(1)
    expect(r.dropped).toEqual({ 'outcome failed': 1, 'context was trimmed': 1, 'nothing recorded': 1 })
    const lines = (await readFile(join(out, 'train.jsonl'), 'utf8')).trim().split('\n')
    const sample = JSON.parse(lines[0])
    expect(sample.messages.map((m: WireMessage) => m.role)).toEqual(['user', 'assistant', 'tool', 'assistant'])
    expect(sample.messages[1].reasoning_content).toBe('think')
    expect(sample.meta).toMatchObject({ conversation: 'good', outcome: 'solved', graded: 'user' })
  })

  it('keeps the fewest-round sessions per practice task', async () => {
    const store = new TraceStore(join(dir, 'data'))
    const sessions: Conversation[] = []
    for (const [id, rounds] of [['a', 3], ['b', 1], ['c', 2]] as const) {
      let msgs: WireMessage[] = [{ role: 'user', content: 'go' }]
      for (let i = 0; i < rounds; i++) {
        await store.record(id, round(`r${i}`, msgs, { toolCalls: [call(`c${i}`, 'rea__open_binary')] }))
        msgs = [...msgs, { role: 'assistant', content: '', tool_calls: [call(`c${i}`, 'rea__open_binary')] }, { role: 'tool', tool_call_id: `c${i}`, content: 'ok' }]
      }
      await store.record(id, round('final', msgs, { content: 'ANSWER: x' }))
      sessions.push(conv(id, { autoGraded: true, practice: { taskId: 't1', family: 'keygen', tier: 'easy' } }))
    }
    const r = await exportDataset(store, sessions, { outDir: join(dir, 'out'), outcomes: ['solved'], skipTrimmed: true, maxPerTask: 2 })
    expect(r.kept).toBe(2)
    const kept = (await readFile(join(dir, 'out', 'train.jsonl'), 'utf8')).trim().split('\n').map((l) => JSON.parse(l).meta.conversation)
    expect(kept).toEqual(['b', 'c'])
  })

  it('rejects conversation ids that could leave the data folder', () => {
    const store = new TraceStore(join(dir, 'data'))
    expect(() => store.record('../x', round('r', [], {}))).toThrow()
  })

  it('lists new rounds as they are recorded, and keeps the trim mark across restarts', async () => {
    const data = join(dir, 'data')
    const store = new TraceStore(data)
    const two: WireMessage[] = [{ role: 'user', content: 'go' }, { role: 'assistant', content: 'a' }]
    await store.record('api-x', round('r1', two, { content: 'one' }), { shrinkIsTrim: true })
    expect(await store.info('api-x')).toMatchObject({ responses: 1, trimmed: false })
    await store.record('api-x', round('r2', [...two, { role: 'user', content: 'more' }], { toolCalls: [call('c1', 'rea__open_binary')] }), { shrinkIsTrim: true })
    expect(await store.info('api-x')).toMatchObject({ responses: 2, toolRounds: 1, trimmed: false })
    // A new store (the app restarted) reads the previous request to notice the shrink.
    const after = new TraceStore(data)
    await after.record('api-x', round('r3', two.slice(0, 1), { content: 'three' }), { shrinkIsTrim: true })
    expect(await after.info('api-x')).toMatchObject({ responses: 3, trimmed: true })
    await after.delete('api-x')
    expect(await after.info('api-x')).toBeNull()
  })

  it('sees a session change even when the file keeps its size and time', async () => {
    const store = new TraceStore(join(dir, 'data'))
    await store.record('api-o', round('r1', [{ role: 'user', content: 'go' }], { content: 'ANSWER: 1' }))
    await store.updateApiSession('api-o', { outcome: 'solved' })
    expect((await store.info('api-o'))?.api?.outcome).toBe('solved')
    // Same length ("solved" / "failed"); the write count changes the cache key whatever the clock says.
    await store.updateApiSession('api-o', { outcome: 'failed' })
    expect((await store.info('api-o'))?.api?.outcome).toBe('failed')
  })

  it('deletes a chat after its pending rounds, and records nothing that arrives later', async () => {
    const store = new TraceStore(join(dir, 'data'))
    const msgs: WireMessage[] = [{ role: 'user', content: 'go' }]
    // Sent without waiting: the delete must still land after the round already queued.
    const first = store.record('chat-d', round('r1', msgs, { content: 'one' }))
    const removed = store.delete('chat-d', undefined, { forever: true })
    await Promise.all([first, removed])
    expect(await store.info('chat-d')).toBeNull()
    // A stopped reply's last round finishing after the delete does not bring the folder back.
    await store.record('chat-d', round('r2', msgs, { content: 'late' }))
    expect(await store.info('chat-d')).toBeNull()
    // Deleting only the trace lets a session that carries on be recorded again.
    await store.record('chat-e', round('r1', msgs, { content: 'one' }))
    await store.delete('chat-e')
    await store.record('chat-e', round('r2', msgs, { content: 'two' }))
    expect(await store.info('chat-e')).toMatchObject({ responses: 1 })
  })

  it('gives an API session default details only when it has none', async () => {
    const store = new TraceStore(join(dir, 'data'))
    // Sent together: the app's own description wins over the defaults however they interleave.
    const described = store.updateApiSession('api-a', { client: 'RulZ Coding Agent', title: 'Find the key' })
    const ensured = store.ensureApiSession('api-a', { client: 'Another app', title: 'a' })
    await Promise.all([described, ensured])
    expect(await store.apiSession('api-a')).toMatchObject({ client: 'RulZ Coding Agent', title: 'Find the key' })
    await store.ensureApiSession('api-b', { client: 'Another app', title: 'b' })
    expect(await store.apiSession('api-b')).toMatchObject({ client: 'Another app', title: 'b', outcome: 'unrated' })
  })

  it('exports API clone sessions with their kind and valid lines whatever the text holds', async () => {
    const store = new TraceStore(join(dir, 'data'))
    await store.record('api-c', round('r1', [{ role: 'user', content: 'clone it' }], { content: 'Done.\nContact:\nme@vendor.com' }))
    const r = await exportDataset(store, [{ id: 'api-c', source: 'RulZ Coding Agent', kind: 'clone-full', re: { target: 'x.exe', goal: 'g', collect: true, outcome: 'solved' } }], {
      outDir: join(dir, 'out'),
      outcomes: ['solved'],
      skipTrimmed: true,
      maxPerTask: 2
    })
    expect(r.kept).toBe(1)
    const sample = JSON.parse((await readFile(join(dir, 'out', 'train.jsonl'), 'utf8')).trim())
    expect(sample.meta).toMatchObject({ source: 'RulZ Coding Agent', kind: 'clone-full' })
    expect(sample.messages.at(-1).content).toBe('Done.\nContact:\nuser@example.com')
  })
})

describe('practice tasks', () => {
  let kit: string
  let work: string
  const task: KitTask = {
    id: 's1-hash_id-000',
    family: 'hash_id',
    tier: 'easy',
    binary: 'prog.exe',
    prompt: 'Which hash does {exe} use?\n\nANSWER: <name>',
    answer: 'crc32',
    check: { type: 'keyword', accept: 'crc32', reject: ['fnv', 'djb2'] }
  }
  beforeEach(async () => {
    kit = await mkdtemp(join(tmpdir(), 'rulz-kit-'))
    work = await mkdtemp(join(tmpdir(), 'rulz-work-'))
    await mkdir(join(kit, 'build', 'bin', task.id), { recursive: true })
    await writeFile(join(kit, 'build', 'bin', task.id, 'prog.exe'), 'MZ')
    await writeFile(join(kit, 'build', 'bin', task.id, 'prog.pdb'), 'pdb')
    await writeFile(join(kit, 'build', 'tasks.jsonl'), JSON.stringify(task) + '\n')
  })
  afterEach(async () => {
    await rm(kit, { recursive: true, force: true })
    await rm(work, { recursive: true, force: true })
  })

  it('lists tasks without their answers', async () => {
    const info = await listPractice(kit)
    expect(info.tasks).toHaveLength(1)
    expect(JSON.stringify(info)).not.toContain('crc32')
    expect((await listPractice(join(kit, 'missing'))).error).toMatch(/kit\.generate/)
  })

  it('copies the binary to a work folder and grades the answer there', async () => {
    const p = await preparePractice(kit, work, task.id)
    expect(p.prompt).toContain(p.target)
    expect(await readFile(p.target, 'utf8')).toBe('MZ')
    expect((await gradePractice(kit, work, task.id, p.target, 'CRC-32')).passed).toBe(true)
    expect((await gradePractice(kit, work, task.id, p.target, 'FNV-1a or CRC32')).passed).toBe(false)
    expect((await gradePractice(kit, work, task.id, p.target, null)).passed).toBe(false)
  })

  it('only grades work-folder copies of the task binary', async () => {
    await expect(gradePractice(kit, work, task.id, join(kit, 'build', 'bin', task.id, 'prog.exe'), 'crc32')).rejects.toThrow(/work-folder/)
    await expect(gradePractice(kit, work, task.id, join(work, task.id, '1', '..', '..', 'prog.exe'), 'crc32')).rejects.toThrow()
  })

  it('checks exact answers', async () => {
    const t: KitTask = { ...task, check: { type: 'exact', value: 'Relay.Example:80', normalize: 'lower' } }
    expect(await checkAnswer(t, 'relay.example:80', '')).toBe(true)
    expect(await checkAnswer({ ...t, check: { type: 'exact', value: 'a' } }, 'A', '')).toBe(false)
  })
})
