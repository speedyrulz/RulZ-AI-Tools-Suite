// WritingRuns: the phone's AI actions as jobs in main, with a fake store, chat and model library (no model, no GPU).
import { describe, expect, it } from 'vitest'
import type { BibleEntry, NovelOutline, NovelProjectState, NovelTextFile, StateOp } from '@shared/novel'
import { defaultProject } from '@shared/novelPrompts'
import { applyResult, hashText, scenePath } from '@shared/novelRun'
import type { WritingJob, WritingRunRequest } from '@shared/phone'
import type { ChatRequest, StreamEvent } from '@shared/types'
import { WritingEvents, writingHandlers } from '../src/main/writing/ipc'
import { WritingRuns, type WritingChat, type WritingModels, type WritingStore } from '../src/main/writing/runs'

const FOLDER = 'C:\\novels\\book'

const entry = (id: string, name: string, extra: Partial<BibleEntry> = {}): BibleEntry => ({
  id,
  name,
  kind: 'character',
  aliases: [],
  triggers: [],
  priority: 50,
  always: false,
  body: `${name} is a person.`,
  problems: [],
  ...extra
})

const scene = (id: string, title: string, extra: Record<string, unknown> = {}) => ({
  id,
  title,
  file: `${id}.md`,
  beats: `Beats of ${title}`,
  summary: '',
  summaryOf: '',
  status: 'beats' as const,
  pov: null,
  present: [] as string[],
  sticky: [] as string[],
  targetWords: 2500 as number | null,
  ...extra
})

const SCENE_TEXT = 'First paragraph.\n\nSecond paragraph about Mara.'

let instanceNo = 0

/** A project on "disk": the state the store returns and the scene files it reads. */
function makeStore(disk: Record<string, string> = { 's1.md': 'The ferry crossed.', 's2.md': SCENE_TEXT }) {
  const outline: NovelOutline = {
    format: 1,
    chapters: [
      { id: 'c1', title: 'One', act: '', summary: '', summaryOf: '', scenes: [scene('s1', 'Ferry')] },
      { id: 'c2', title: 'Two', act: '', summary: '', summaryOf: '', scenes: [scene('s2', 'Dock')] }
    ]
  }
  const project = defaultProject('Book')
  project.sampling = { ...project.sampling, seed: 100 }
  const state: NovelProjectState = {
    folder: FOLDER,
    project,
    outline,
    bible: [entry('mara', 'Mara', { always: true }), entry('oskar', 'Oskar'), entry('dock', 'The Dock', { kind: 'place' })],
    style: { path: 'style.md', text: 'Dry and plain.', hash: 'sh', meta: {} as NovelTextFile['meta'], readOnly: false, journal: null },
    words: {},
    missing: [],
    orphans: [],
    blocked: null,
    state: { facts: [], invalid: [], problem: null, canReset: false },
    dismissed: {},
    checksProblem: null
  }
  const patches: StateOp[][] = []
  const held: string[] = []
  const released: string[] = []
  const closed = { value: false }
  const store: WritingStore = {
    hold: async (_f, owner) => void held.push(owner),
    release: async (_f, owner) => void released.push(owner),
    projectState: async () => state,
    readFile: async (_f, rel) => {
      if (closed.value) throw new Error('That project is not open.')
      const name = rel.replace('manuscript/', '')
      if (!(name in disk)) throw new Error('missing')
      return { path: rel, text: disk[name], hash: `hash:${disk[name]}`, meta: {} as NovelTextFile['meta'], readOnly: false, journal: null }
    },
    statePatch: async (_f, ops) => {
      patches.push(ops)
      return { state, rejected: [] }
    }
  }
  return { store, state, disk, patches, held, released, closed }
}

/** A chat whose replies are scripted per request. `hold` keeps the stream open until released or aborted. */
function makeChat(reply: (req: ChatRequest, n: number) => string | { chunks: string[]; hold?: Promise<void> }) {
  const handlers = new Map<string, (e: StreamEvent) => void>()
  const requests: ChatRequest[] = []
  const aborted: string[] = []
  const aborts = new Map<string, () => void>()
  const chat: WritingChat = {
    countText: async (_m, text) => Math.ceil(text.length / 4),
    countTokens: async (_m, messages) => messages.reduce((n, m) => n + Math.ceil(String(m.content).length / 4) + 4, 0),
    start: async (req) => {
      requests.push(req)
      const send = (e: StreamEvent): void => handlers.get(req.requestId)?.(e)
      const r = reply(req, requests.length - 1)
      const spec = typeof r === 'string' ? { chunks: [r], hold: undefined } : r
      send({ type: 'progress', requestId: req.requestId, processed: 5, total: 10 })
      for (const c of spec.chunks) send({ type: 'delta', requestId: req.requestId, content: c })
      let stopped = false
      if (spec.hold) {
        await Promise.race([spec.hold, new Promise<void>((res) => aborts.set(req.requestId, () => ((stopped = true), res())))])
      }
      send({ type: 'done', requestId: req.requestId, stats: { stopReason: stopped ? 'stopped' : 'stop' } })
    },
    abort: (id) => {
      aborted.push(id)
      aborts.get(id)?.()
    },
    subscribe: (id, h) => {
      handlers.set(id, h)
      return () => void handlers.delete(id)
    }
  }
  return { chat, requests, aborted }
}

function setup(opts: { disk?: Record<string, string>; reply?: Parameters<typeof makeChat>[0]; busy?: () => string | null; throttleMs?: number; max?: number } = {}) {
  const s = makeStore(opts.disk)
  const c = makeChat(opts.reply ?? (() => 'Generated text.'))
  const models: WritingModels = {
    list: () => [
      { id: 'm1', name: 'Model One' },
      { id: 'm2', name: 'Model Two' }
    ],
    ready: (id) => (id === null || id === 'm1' ? { modelId: 'm1', instanceId: `inst-${++instanceNo}`, contextLength: 8192 } : undefined),
    ensure: async () => {
      throw new Error('no GPU in tests')
    }
  }
  const events: WritingJob[] = []
  let ids = 0
  const runs = new WritingRuns({
    store: s.store,
    chat: c.chat,
    models,
    emit: (j) => events.push(j),
    busyReason: opts.busy,
    newId: () => `job${++ids}`,
    throttleMs: opts.throttleMs ?? 250,
    maxJobsPerFolder: opts.max
  })
  return { ...s, ...c, runs, events, models }
}

const request = (over: Partial<WritingRunRequest> = {}): WritingRunRequest => ({
  folder: FOLDER,
  action: 'continue',
  sceneId: 's2',
  text: SCENE_TEXT,
  baseHash: `hash:${SCENE_TEXT}`,
  cursor: SCENE_TEXT.length,
  selection: null,
  note: '',
  ...over
})

const RUNNING = ['queued', 'preparing', 'waiting', 'writing']

async function finished(runs: WritingRuns, id: string): Promise<WritingJob> {
  for (let i = 0; i < 2000; i++) {
    const j = runs.get(id)
    if (j && !RUNNING.includes(j.status)) return j
    await new Promise((r) => setTimeout(r, 2))
  }
  throw new Error('the job did not finish')
}

async function waitFor(cond: () => boolean): Promise<void> {
  for (let i = 0; i < 2000 && !cond(); i++) await new Promise((r) => setTimeout(r, 2))
  if (!cond()) throw new Error('condition not met')
}

describe('prose actions', () => {
  it('writes two alternatives whose texts are what the desktop would apply', async () => {
    const t = setup({ reply: (_r, n) => (n === 0 ? '  Alpha continues.  ' : 'Beta continues.') })
    const job = t.runs.run(request({ alternatives: 2 }))
    expect(job.status).toBe('preparing')
    expect(job.of).toBe(2)
    expect(job.request).not.toHaveProperty('text')
    const done = await finished(t.runs, job.id)
    expect(done.status).toBe('done')
    expect(done.result).toMatchObject({ kind: 'prose' })
    if (done.result?.kind !== 'prose') throw new Error('prose expected')
    expect(done.result.alts.map((a) => a.generated)).toEqual(['Alpha continues.', 'Beta continues.'])
    expect(done.result.alts.map((a) => a.text)).toEqual([
      applyResult('continue', SCENE_TEXT, SCENE_TEXT.length, null, 'Alpha continues.'),
      applyResult('continue', SCENE_TEXT, SCENE_TEXT.length, null, 'Beta continues.')
    ])
    expect(done.result.alts.map((a) => a.seed)).toEqual([100, 101])
    // Mara is a fixed entry the text mentions: she is in the request; Oskar is not.
    const user = String(t.requests[0].messages[1].content)
    expect(user).toContain('Mara')
    expect(t.requests[0]).toMatchObject({ prefitted: true, overflow: 'stop', disableThinking: true })
    expect(t.requests[0].sampling).toMatchObject({ seed: 100, thinking: 'off', jsonSchema: '' })
    expect(t.requests[1].sampling.seed).toBe(101)
    // Status phases, in order, without repeats of the final state.
    const phases = t.events.map((e) => e.status)
    expect(phases[0]).toBe('preparing')
    expect(phases).toContain('waiting')
    expect(phases).toContain('writing')
    expect(phases[phases.length - 1]).toBe('done')
    expect(t.events.some((e) => e.alt === 2)).toBe(true)
    expect(done.partial).toBe('')
    expect(done.endedAt).not.toBeNull()
  })

  it('replaces the selection for a rewrite and refuses a rewrite without one', async () => {
    const t = setup({ reply: () => 'Rewritten.' })
    const bad = t.runs.run(request({ action: 'rewrite' }))
    expect((await finished(t.runs, bad.id)).error).toBe('Select the text to change first.')
    const job = t.runs.run(request({ action: 'rewrite', variant: 'tighter', selection: { from: 0, to: 16 } }))
    expect(job.label).toBe('Rewrite: tighter')
    const done = await finished(t.runs, job.id)
    if (done.result?.kind !== 'prose') throw new Error('prose expected')
    expect(done.result.alts[0].text).toBe(`Rewritten.${SCENE_TEXT.slice(16)}`)
  })

  it('keeps the alternatives written when a later one fails', async () => {
    const t = setup({
      reply: (_r, n) => {
        if (n === 1) throw new Error('boom')
        return 'First.'
      }
    })
    const job = t.runs.run(request({ alternatives: 2 }))
    const done = await finished(t.runs, job.id)
    expect(done.status).toBe('done')
    expect(done.warnings.join(' ')).toContain('Stopped after 1 of 2')
    expect(done.result?.kind === 'prose' && done.result.alts).toHaveLength(1)
  })

  it('fails with a plain sentence when no model is ready and the project has none', async () => {
    const t = setup()
    t.models.ready = () => undefined
    const job = t.runs.run(request())
    const done = await finished(t.runs, job.id)
    expect(done.status).toBe('error')
    expect(done.error).toContain('No model is loaded')
  })

  it('tells the phone to load the model when the project\'s own model cannot be loaded', async () => {
    const t = setup()
    t.state.project.modelId = 'm2'
    const job = t.runs.run(request())
    const done = await finished(t.runs, job.id)
    expect(done.status).toBe('error')
    expect(done.error).toContain('Load it in LLM Suite first')
  })
})

describe('structured actions', () => {
  it('summarizes a scene and ties the summary to the disk hash', async () => {
    const t = setup({ reply: () => '{"summary":"Mara reaches the dock."}' })
    const done = await finished(t.runs, t.runs.run(request({ action: 'summarizeScene' })).id)
    expect(done.result).toEqual({ kind: 'summary', target: 'scene', targetId: 's2', reply: { summary: 'Mara reaches the dock.' }, summaryOf: `hash:${SCENE_TEXT}` })
    expect(t.requests[0].sampling.jsonSchema).not.toBe('')
    expect(t.requests[0].disableThinking).toBe(false)
  })

  it('marks a summary out of date when the phone text has unsaved edits', async () => {
    const t = setup({ reply: () => '{"summary":"x"}' })
    const done = await finished(t.runs, t.runs.run(request({ action: 'summarizeScene', text: `${SCENE_TEXT} More.` })).id)
    expect(done.result).toMatchObject({ kind: 'summary', summaryOf: '' })
    expect(done.warnings.join(' ')).toContain('not saved')
  })

  it('summarizes a chapter with the chapter hash key', async () => {
    const t = setup({ reply: () => '{"summary":"Chapter one."}' })
    const done = await finished(t.runs, t.runs.run(request({ action: 'summarizeChapter', sceneId: null, chapterId: 'c1', text: '' })).id)
    expect(done.result).toMatchObject({ kind: 'summary', target: 'chapter', targetId: 'c1' })
  })

  it('proposes scenes, maps names to bible ids and lists the names that match nothing', async () => {
    const reply = JSON.stringify({ scenes: [{ title: 'Storm', beats: 'A storm hits.', pov: 'Mara', present: ['Mara', 'Oskar', 'Nobody'] }] })
    const t = setup({ reply: () => reply })
    const done = await finished(t.runs, t.runs.run(request({ action: 'outlineScenes', sceneId: null, chapterId: 'c2', text: '', count: 1 })).id)
    expect(done.result).toEqual({
      kind: 'scenes',
      chapterId: 'c2',
      reply: { scenes: [{ title: 'Storm', beats: 'A storm hits.', pov: 'mara', present: ['mara', 'oskar'] }] },
      unresolved: ['Nobody']
    })
  })

  it('proposes chapters', async () => {
    const t = setup({ reply: () => '```json\n{"chapters":[{"title":"Start","act":"I","summary":"It begins."}]}\n```' })
    const done = await finished(t.runs, t.runs.run(request({ action: 'outlineChapters', sceneId: null, text: '' })).id)
    expect(done.result).toEqual({ kind: 'chapters', reply: { chapters: [{ title: 'Start', act: 'I', summary: 'It begins.' }] }, unresolved: [] })
  })

  it('reports a reply that is not JSON as an error', async () => {
    const t = setup({ reply: () => 'I could not do that.' })
    const done = await finished(t.runs, t.runs.run(request({ action: 'summarizeScene' })).id)
    expect(done.status).toBe('error')
    expect(done.error).toMatch(/not valid JSON/)
  })

  it('checks a scene and drops repeated findings', async () => {
    const f = { type: 'continuity', quote: 'Second paragraph', problem: 'Mara was elsewhere.', conflictsWith: '', suggestion: 'Move her.' }
    const t = setup({ reply: () => JSON.stringify({ findings: [f, { ...f, problem: 'again' }, { type: 'tense', quote: 'First', problem: 'tense slip', conflictsWith: '', suggestion: '' }] }) })
    const done = await finished(t.runs, t.runs.run(request({ action: 'checkScene' })).id)
    expect(done.result?.kind).toBe('check')
    if (done.result?.kind !== 'check') throw new Error('check expected')
    expect(done.result.sceneId).toBe('s2')
    expect(done.result.findings.map((x) => x.type)).toEqual(['continuity', 'tense'])
    expect(t.requests[0].sampling.jsonSchema).not.toBe('')
  })

  it('does not check a whole chapter in one job', async () => {
    const t = setup()
    const done = await finished(t.runs, t.runs.run(request({ action: 'checkScene', sceneId: null, chapterId: 'c1' })).id)
    expect(done.status).toBe('error')
    expect(done.error).toContain('one scene at a time')
  })
})

describe('update state', () => {
  const reply = JSON.stringify({
    changes: [
      { op: 'add', entity: 'Mara', kind: 'carries', value: 'the brass key', about: null },
      { op: 'add', entity: 'Nobody', kind: 'location', value: 'the moon', about: null },
      { op: 'end', factId: 'nope', reason: 'x' }
    ]
  })

  it('resolves changes against the bible and applies the chosen ones when the scene is unchanged', async () => {
    const t = setup({ reply: () => reply })
    const done = await finished(t.runs, t.runs.run(request({ action: 'updateState' })).id)
    if (done.result?.kind !== 'state') throw new Error('state expected')
    expect(done.result.textHash).toBe(hashText(SCENE_TEXT))
    expect(done.result.items).toHaveLength(1)
    expect(done.result.items[0]).toMatchObject({ op: 'add', fact: { entity: 'mara', kind: 'carries', value: 'the brass key', about: null, from: 's2', until: null } })
    expect(done.result.left).toHaveLength(2)
    const res = await t.runs.applyState(done.id, [done.result.items[0].key])
    expect(res.rejected).toEqual([])
    expect(t.patches).toHaveLength(1)
    expect(t.patches[0][0]).toMatchObject({ op: 'add', fact: { entity: 'mara', value: 'the brass key', from: 's2' } })
    expect((t.patches[0][0] as Extract<StateOp, { op: 'add' }>).fact.id).toMatch(/^f/)
    // The same item cannot be applied twice.
    await expect(t.runs.applyState(done.id, [done.result.items[0].key])).rejects.toThrow('Nothing is accepted')
  })

  it('refuses to apply when the scene text on disk changed', async () => {
    const t = setup({ reply: () => reply })
    const done = await finished(t.runs, t.runs.run(request({ action: 'updateState' })).id)
    if (done.result?.kind !== 'state') throw new Error('state expected')
    t.disk['s2.md'] = `${SCENE_TEXT} Edited elsewhere.`
    await expect(t.runs.applyState(done.id, [done.result.items[0].key])).rejects.toThrow('The scene text changed since this was made')
    expect(t.patches).toHaveLength(0)
  })

  it('refuses a job that is not a state update', async () => {
    const t = setup({ reply: () => '{"summary":"x"}' })
    const done = await finished(t.runs, t.runs.run(request({ action: 'summarizeScene' })).id)
    await expect(t.runs.applyState(done.id, ['0'])).rejects.toThrow('not a finished state update')
  })

  it('refuses when state.json cannot be used', async () => {
    const t = setup({ reply: () => reply })
    t.state.state = { facts: [], invalid: [], problem: 'state.json is damaged.', canReset: false }
    const done = await finished(t.runs, t.runs.run(request({ action: 'updateState' })).id)
    expect(done.error).toContain('state.json cannot be used')
  })
})

describe('apply state, concurrency and rejections', () => {
  const reply = JSON.stringify({
    changes: [
      { op: 'add', entity: 'Mara', kind: 'carries', value: 'the brass key', about: null },
      { op: 'add', entity: 'Oskar', kind: 'location', value: 'the dock', about: null }
    ]
  })

  it('applies a proposal once when two applyState calls arrive together', async () => {
    const t = setup({ reply: () => reply })
    const done = await finished(t.runs, t.runs.run(request({ action: 'updateState' })).id)
    if (done.result?.kind !== 'state') throw new Error('state expected')
    const keys = done.result.items.map((i) => i.key)
    const [a, b] = await Promise.allSettled([t.runs.applyState(done.id, keys), t.runs.applyState(done.id, keys)])
    expect(a.status).toBe('fulfilled')
    expect(b.status).toBe('rejected')
    expect(t.patches).toHaveLength(1)
    expect(t.patches[0]).toHaveLength(2)
  })

  it('keeps a rejected item applicable and marks only the written ones', async () => {
    const t = setup({ reply: () => reply })
    const done = await finished(t.runs, t.runs.run(request({ action: 'updateState' })).id)
    if (done.result?.kind !== 'state') throw new Error('state expected')
    const keys = done.result.items.map((i) => i.key)
    const patch = t.store.statePatch
    t.store.statePatch = async (_f, ops) => {
      t.patches.push(ops)
      return { state: t.state, rejected: [{ index: 1, reason: 'no' }] }
    }
    const first = await t.runs.applyState(done.id, keys)
    expect(first.rejected).toHaveLength(1)
    t.store.statePatch = patch
    await t.runs.applyState(done.id, keys)
    expect(t.patches).toHaveLength(2)
    expect(t.patches[1]).toHaveLength(1)
    expect(t.patches[1][0]).toMatchObject({ op: 'add', fact: { value: 'the dock' } })
  })
})

describe('job ownership', () => {
  const holds = new Map<string, Set<string>>()
  const own = (reply?: Parameters<typeof makeChat>[0]) => {
    holds.clear()
    holds.set('phone:a', new Set([FOLDER]))
    const t = setup({ reply })
    const runs = new WritingRuns({ store: t.store, chat: t.chat, models: t.models, emit: () => undefined, newId: () => 'j1', holds: (o, f) => holds.get(o)?.has(f) === true })
    return { t, runs }
  }

  it('lets the starting owner and any owner holding the folder use a job, and refuses others', async () => {
    const { runs } = own()
    expect(() => runs.run(request(), 'phone:x')).toThrow('not open on this phone')
    const job = runs.run(request(), 'phone:a')
    await finished(runs, job.id)
    expect(runs.get(job.id, 'phone:a')?.status).toBe('done')
    expect(runs.list(FOLDER, 'phone:a')).toHaveLength(1)
    // A phone that reconnected has a new id but holds the folder: it gets its job back.
    holds.set('phone:b', new Set([FOLDER]))
    expect(runs.get(job.id, 'phone:b')?.status).toBe('done')
    // A phone that holds nothing is refused.
    expect(() => runs.get(job.id, 'phone:c')).toThrow('That job belongs to another phone.')
    expect(() => runs.abort(job.id, 'phone:c')).toThrow('That job belongs to another phone.')
    expect(() => runs.dismiss(job.id, 'phone:c')).toThrow('That job belongs to another phone.')
    await expect(runs.applyState(job.id, ['0'], 'phone:c')).rejects.toThrow('That job belongs to another phone.')
    expect(() => runs.list(FOLDER, 'phone:c')).toThrow('not open on this phone')
    // The owner that started it still may after releasing the folder.
    holds.delete('phone:a')
    expect(runs.get(job.id, 'phone:a')?.id).toBe(job.id)
    runs.dismiss(job.id, 'phone:b')
    expect(runs.get(job.id, 'phone:a')).toBeNull()
  })
})

describe('project hold, dismiss and request building', () => {
  it('holds the project for the job and releases it at the end', async () => {
    const t = setup()
    const job = t.runs.run(request())
    await finished(t.runs, job.id)
    await waitFor(() => t.released.length === 1)
    expect(t.held).toEqual([`job:${job.id}`])
    expect(t.released).toEqual([`job:${job.id}`])
  })

  it('fails the job when the project is not open any more instead of going on with missing data', async () => {
    const t = setup()
    t.closed.value = true
    const done = await finished(t.runs, t.runs.run(request()).id)
    expect(done.status).toBe('error')
    expect(done.error).toContain('not open')
  })

  it('fails the job when the project cannot be held, and releases nothing', async () => {
    const t = setup()
    t.store.hold = async () => {
      throw new Error('That folder does not exist.')
    }
    const done = await finished(t.runs, t.runs.run(request()).id)
    expect(done.error).toBe('The project could not be opened: That folder does not exist.')
    expect(t.released).toEqual([])
  })

  it('still counts a dismissed running job as the running one', async () => {
    let release: () => void = () => undefined
    const hold = new Promise<void>((r) => (release = r))
    const t = setup({ reply: () => ({ chunks: ['x'], hold }) })
    const job = t.runs.run(request())
    await waitFor(() => t.requests.length === 1)
    t.runs.dismiss(job.id)
    expect(t.runs.get(job.id)).toBeNull()
    expect(() => t.runs.run(request())).toThrow('already working on another AI action')
    await waitFor(() => t.released.length === 1)
    expect(() => t.runs.run(request())).not.toThrow()
    release()
  })

  it('builds chapter actions for the chapter even when the phone also sends a scene', async () => {
    const t = setup({ reply: () => JSON.stringify({ scenes: [{ title: 'A', beats: 'b', pov: null, present: [] }] }) })
    const done = await finished(t.runs, t.runs.run(request({ action: 'outlineScenes', sceneId: 's1', chapterId: 'c2', text: 'ignored', count: 1 })).id)
    expect(done.status).toBe('done')
    expect(String(t.requests[0].messages[1].content)).toContain('Chapter 2: Two')
    const t2 = setup({ reply: () => '{"summary":"x"}' })
    await finished(t2.runs, t2.runs.run(request({ action: 'summarizeChapter', sceneId: 's1', chapterId: 'c2', text: '' })).id)
    expect(String(t2.requests[0].messages[1].content)).toContain('Chapter: Two')
  })

  it('sends the whole-scene actions without a note or selection, as the desktop does', async () => {
    for (const action of ['summarizeScene', 'checkScene', 'updateState'] as const) {
      const t = setup({ reply: () => '{"summary":"x","findings":[],"changes":[]}' })
      await finished(t.runs, t.runs.run(request({ action, note: 'zzz-note', cursor: 0, selection: { from: 0, to: 5 } })).id)
      expect(t.requests).toHaveLength(1)
      expect(JSON.stringify(t.requests[0].messages)).not.toContain('zzz-note')
    }
    const t = setup()
    await finished(t.runs, t.runs.run(request({ note: 'zzz-note' })).id)
    expect(JSON.stringify(t.requests[0].messages)).toContain('zzz-note')
  })
})

describe('interview', () => {
  it('answers the last question with the earlier turns as history', async () => {
    const t = setup({ reply: () => '  I was afraid.  ' })
    const done = await finished(
      t.runs,
      t.runs.run(
        request({
          action: 'interview',
          text: '',
          interview: {
            characterId: 'mara',
            turns: [
              { role: 'user', text: 'Who are you?' },
              { role: 'assistant', text: 'Mara.' },
              { role: 'user', text: 'Were you afraid?' }
            ]
          }
        })
      ).id
    )
    expect(done.result).toEqual({ kind: 'interview', characterId: 'mara', reply: 'I was afraid.' })
    const msgs = t.requests[0].messages.map((m) => `${m.role}:${m.content}`)
    expect(msgs.slice(-3)).toEqual(['user:Who are you?', 'assistant:Mara.', 'user:Were you afraid?'])
    expect(t.requests[0].messages[0].role).toBe('system')
    expect(t.requests[0]).toMatchObject({ disableThinking: true })
  })

  it('needs a known character and a question', async () => {
    const t = setup()
    const a = await finished(t.runs, t.runs.run(request({ action: 'interview', interview: { characterId: 'ghost', turns: [{ role: 'user', text: 'Hi' }] } })).id)
    expect(a.error).toContain('not in the bible')
    const b = await finished(t.runs, t.runs.run(request({ action: 'interview', interview: { characterId: 'mara', turns: [] } })).id)
    expect(b.error).toContain('Ask a question')
  })
})

describe('stopping, refusing, events and housekeeping', () => {
  it('stops a running job and keeps nothing when no text was written', async () => {
    let release: () => void = () => undefined
    const hold = new Promise<void>((r) => (release = r))
    const t = setup({ reply: () => ({ chunks: [], hold }) })
    const job = t.runs.run(request())
    await waitFor(() => t.requests.length === 1)
    t.runs.abort(job.id)
    const done = await finished(t.runs, job.id)
    release()
    expect(done.status).toBe('stopped')
    expect(done.result).toBeNull()
    expect(t.aborted).toEqual([t.requests[0].requestId])
  })

  it('keeps the text written before the stop as an alternative', async () => {
    const hold = new Promise<void>(() => undefined)
    const t = setup({ reply: () => ({ chunks: ['Half a thought'], hold }) })
    const job = t.runs.run(request({ alternatives: 3 }))
    await waitFor(() => t.requests.length === 1 && (t.runs.get(job.id)?.partial ?? '') !== '')
    t.runs.abort(job.id)
    const done = await finished(t.runs, job.id)
    expect(done.status).toBe('stopped')
    expect(done.result?.kind === 'prose' && done.result.alts.map((a) => a.generated)).toEqual(['Half a thought'])
    expect(t.requests).toHaveLength(1)
  })

  it('refuses a second job while one runs, and takes one after it ends', async () => {
    let release: () => void = () => undefined
    const hold = new Promise<void>((r) => (release = r))
    const t = setup({ reply: () => ({ chunks: ['x'], hold }) })
    const first = t.runs.run(request())
    expect(() => t.runs.run(request())).toThrow('already working on another AI action from a phone')
    release()
    await finished(t.runs, first.id)
    expect(() => t.runs.run(request())).not.toThrow()
  })

  it('refuses while the desktop is generating', () => {
    const t = setup({ busy: () => 'LLM Suite is writing something else on the computer.' })
    expect(() => t.runs.run(request())).toThrow('LLM Suite is writing something else on the computer. Try again when it has finished.')
    expect(t.runs.list(FOLDER)).toEqual([])
  })

  it('sends status changes at once and streaming text at most once per throttle gap', async () => {
    let release: () => void = () => undefined
    const hold = new Promise<void>((r) => (release = r))
    const chunks = Array.from({ length: 40 }, (_, i) => `w${i} `)
    const t = setup({ throttleMs: 60, reply: () => ({ chunks, hold }) })
    const job = t.runs.run(request())
    await waitFor(() => t.runs.get(job.id)?.partial.startsWith('w0') === true && t.runs.get(job.id)!.partial.includes('w39'))
    // 40 deltas arrived in one tick: the status change to 'writing' went out, the text follows after the gap.
    const writing = (): WritingJob[] => t.events.filter((e) => e.status === 'writing')
    expect(writing().length).toBeLessThanOrEqual(2)
    await waitFor(() => writing().some((e) => e.partial.includes('w39')))
    const gapEmits = writing().length
    expect(gapEmits).toBeLessThanOrEqual(3)
    release()
    await finished(t.runs, job.id)
    expect(t.events[t.events.length - 1].status).toBe('done')
    expect(t.events.filter((e) => e.status === 'waiting').length).toBeLessThanOrEqual(2)
  })

  it('lists jobs per folder and forgets dismissed ones', async () => {
    const t = setup()
    const a = t.runs.run(request())
    await finished(t.runs, a.id)
    const b = t.runs.run(request({ folder: 'C:\\novels\\other' }))
    await finished(t.runs, b.id)
    expect(t.runs.list(FOLDER).map((j) => j.id)).toEqual([a.id])
    expect(t.runs.get(a.id)?.status).toBe('done')
    t.runs.dismiss(a.id)
    expect(t.runs.get(a.id)).toBeNull()
    expect(t.runs.list(FOLDER)).toEqual([])
    expect(t.runs.list('C:\\novels\\other')).toHaveLength(1)
    // Dismissing something unknown is harmless.
    t.runs.dismiss('nope')
  })

  it('keeps at most the cap per folder, dropping the oldest finished', async () => {
    const t = setup({ max: 3 })
    const ids: string[] = []
    for (let i = 0; i < 5; i++) {
      const j = t.runs.run(request())
      ids.push(j.id)
      await finished(t.runs, j.id)
    }
    expect(t.runs.list(FOLDER).map((j) => j.id)).toEqual(ids.slice(-3))
  })

  it('lists the model library with what is loaded', () => {
    const t = setup()
    expect(t.runs.models()).toEqual([
      { id: 'm1', name: 'Model One', loaded: true },
      { id: 'm2', name: 'Model Two', loaded: false }
    ])
  })

  it('refuses a read-only project and an unknown scene', async () => {
    const t = setup()
    t.state.blocked = 'outline.json could not be read.'
    expect((await finished(t.runs, t.runs.run(request()).id)).error).toContain('read-only')
    t.state.blocked = null
    expect((await finished(t.runs, t.runs.run(request({ sceneId: 'gone' })).id)).error).toBe('The scene is not in the outline any more.')
  })
})

describe('draft scene on an empty scene', () => {
  it('leads in from the previous scene on disk', async () => {
    const t = setup({ disk: { 's1.md': 'The ferry crossed at dawn.', 's2.md': '' }, reply: () => 'A new scene begins.' })
    const done = await finished(t.runs, t.runs.run(request({ action: 'draftScene', text: '', cursor: 0, baseHash: 'hash:' })).id)
    if (done.result?.kind !== 'prose') throw new Error('prose expected')
    expect(done.result.alts[0].text).toBe('A new scene begins.')
    expect(String(t.requests[0].messages[1].content)).toContain('The ferry crossed at dawn.')
    expect(scenePath({ file: 's2.md' })).toBe('manuscript/s2.md')
  })
})

describe('phone handlers', () => {
  it('answers every writing channel with checked arguments and fans job events out', async () => {
    const events = new WritingEvents()
    const seen: string[] = []
    const off = events.subscribe((j) => seen.push(j.status))
    events.subscribe(() => {
      throw new Error('a broken listener')
    })
    const t = setup()
    const runs = new WritingRuns({ store: t.store, chat: t.chat, models: t.models, emit: events.emit, newId: () => 'j1', holds: (o, folder) => o === 'phone:1' && folder === FOLDER })
    const h = writingHandlers(runs)
    expect(Object.keys(h).sort()).toEqual(['writing:abort', 'writing:applyState', 'writing:dismiss', 'writing:job', 'writing:jobs', 'writing:models', 'writing:run'])
    const job = (await h['writing:run']('phone:1', request())) as WritingJob
    expect(job.id).toBe('j1')
    await finished(runs, 'j1')
    expect(seen[0]).toBe('preparing')
    expect(seen[seen.length - 1]).toBe('done')
    expect(((await h['writing:jobs']('phone:1', FOLDER)) as WritingJob[]).map((j) => j.id)).toEqual(['j1'])
    expect(((await h['writing:job']('phone:1', 'j1')) as WritingJob).status).toBe('done')
    expect(await h['writing:job']('phone:1', 'zzz')).toBeNull()
    await expect(h['writing:jobs']('phone:1', 5)).rejects.toThrow('project folder is missing')
    expect(await h['writing:models']('phone:1')).toHaveLength(2)
    await h['writing:abort']('phone:1', 'j1')
    await h['writing:dismiss']('phone:1', 'j1')
    expect(await h['writing:job']('phone:1', 'j1')).toBeNull()
    off()
  })
})
