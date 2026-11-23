// AdventureRuns with a fake chat and model library and a real store on a temp folder (no model, no GPU): the order
// of saving, stop and failure at each step, retries, the model gate, throttled events and context fitting.
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { noChanges, type AdventureJob, type AdventureSetup } from '../src/shared/adventure'
import { DEFAULT_SAMPLING } from '../src/shared/defaults'
import { LEFT_OUT_LINE } from '../src/shared/adventurePrompts'
import type { ChatRequest, StreamEvent } from '../src/shared/types'
import { AdventureRuns } from '../src/main/adventure/runs'
import { AdventureStore } from '../src/main/adventure/store'
import { ModelGate } from '../src/main/modelGate'
import { WritingRuns, type WritingChat, type WritingModels, type WritingStore } from '../src/main/writing/runs'

const dirs: string[] = []
afterEach(() => {
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true })
})

const CHOICES_JSON = JSON.stringify({
  choices: ['Open the door', 'Run away', 'Call out'],
  ended: false,
  changes: { addItems: ['lamp'], removeItems: [], setStats: [{ name: 'Health', value: 10 }], removeStats: [], addFlags: ['at the door'], removeFlags: [] }
})

type Reply = string | { chunks: string[]; hold?: Promise<void> }

/** A chat whose replies are scripted per request. Passage requests have no schema; choosing requests have one. */
function makeChat(reply: (req: ChatRequest, n: number, isChoosing: boolean) => Reply) {
  const handlers = new Map<string, (e: StreamEvent) => void>()
  const requests: ChatRequest[] = []
  const aborts = new Map<string, () => void>()
  const chat: WritingChat = {
    countText: async (_m, text) => Math.ceil(text.length / 4),
    countTokens: async (_m, messages) => messages.reduce((n, m) => n + Math.ceil(String(m.content).length / 4) + 4, 0),
    start: async (req) => {
      requests.push(req)
      const send = (e: StreamEvent): void => handlers.get(req.requestId)?.(e)
      const r = reply(req, requests.length - 1, req.sampling.jsonSchema !== '')
      const spec = typeof r === 'string' ? { chunks: [r], hold: undefined } : r
      send({ type: 'progress', requestId: req.requestId, processed: 5, total: 10 })
      for (const c of spec.chunks) send({ type: 'delta', requestId: req.requestId, content: c })
      let stopped = false
      if (spec.hold) await Promise.race([spec.hold, new Promise<void>((res) => aborts.set(req.requestId, () => ((stopped = true), res())))])
      send({ type: 'done', requestId: req.requestId, stats: { stopReason: stopped ? 'stopped' : 'stop' } })
    },
    abort: (id) => aborts.get(id)?.(),
    subscribe: (id, h) => {
      handlers.set(id, h)
      return () => void handlers.delete(id)
    }
  }
  return { chat, requests }
}

interface Opts {
  reply?: Parameters<typeof makeChat>[0]
  busy?: () => string | null
  throttleMs?: number
  maxFinished?: number
  contextLength?: number | null
  setup?: Partial<AdventureSetup>
}

async function setup(opts: Opts = {}) {
  const dir = mkdtempSync(join(tmpdir(), 'adv-runs-'))
  dirs.push(dir)
  let ids = 0
  const store = new AdventureStore({ dir: join(dir, 'adventures'), newId: () => `id-${String(++ids).padStart(8, '0')}` })
  const c = makeChat(opts.reply ?? ((_r, _n, choosing) => (choosing ? CHOICES_JSON : 'You stand at a gate. What now?')))
  const ensured: string[] = []
  const models: WritingModels = {
    list: () => [
      { id: 'm1', name: 'Model One' },
      { id: 'm2', name: 'Model Two' }
    ],
    ready: (id) => (id === null || id === 'm1' ? { modelId: 'm1', instanceId: 'inst-1', contextLength: opts.contextLength === undefined ? 8192 : opts.contextLength } : undefined),
    ensure: async (id) => {
      ensured.push(id)
      if (id === 'm2') return { modelId: 'm2', instanceId: 'inst-2', contextLength: 8192 }
      throw new Error('no GPU in tests')
    }
  }
  const gate = new ModelGate()
  const events: AdventureJob[] = []
  const runs = new AdventureRuns({
    store,
    chat: c.chat,
    models,
    gate,
    emit: (j) => events.push(j),
    busyReason: opts.busy,
    newId: () => `job${++ids}`.padEnd(8, '0'),
    seed: () => 7,
    throttleMs: opts.throttleMs ?? 250,
    maxFinishedPerAdventure: opts.maxFinished
  })
  const adv = await store.create({ title: 'Lantern', genre: 'Fantasy', premise: 'A keeper.', character: 'Wren', tone: 'Dry', length: 'short', choiceCount: 3, modelId: null, temperature: null, ...opts.setup })
  return { ...c, store, runs, gate, events, models, ensured, adv, advId: adv.id }
}

const RUNNING = ['queued', 'preparing', 'waiting', 'writing', 'choosing']

async function finished(runs: AdventureRuns, advId: string, jobId: string): Promise<AdventureJob> {
  for (let i = 0; i < 2000; i++) {
    const j = runs.list(advId).find((x) => x.id === jobId)
    if (j && !RUNNING.includes(j.status)) return j
    await new Promise((r) => setTimeout(r, 2))
  }
  throw new Error('the job did not finish')
}

async function waitFor(cond: () => boolean | Promise<boolean>): Promise<void> {
  for (let i = 0; i < 2000 && !(await cond()); i++) await new Promise((r) => setTimeout(r, 2))
  if (!(await cond())) throw new Error('condition not met')
}

async function opening(t: Awaited<ReturnType<typeof setup>>): Promise<AdventureJob> {
  const job = await t.runs.play({ adventureId: t.advId, parentId: null, input: { kind: 'start' } })
  return finished(t.runs, t.advId, job.id)
}

describe('the opening', () => {
  it('writes the passage, saves it, then fills in the choices and state', async () => {
    const t = await setup()
    const job = await t.runs.play({ adventureId: t.advId, parentId: null, input: { kind: 'start' } })
    expect(job.status).toBe('preparing')
    expect(job.kind).toBe('play')
    expect(job.nodeId).toBeNull()
    const done = await finished(t.runs, t.advId, job.id)
    expect(done.status).toBe('done')
    expect(done.error).toBeNull()
    expect(done.endedAt).not.toBeNull()
    expect(done.progress).toBeNull()
    const adv = await t.store.get(t.advId)
    const node = adv.nodes[done.resultNodeId as string]
    expect(adv.rootId).toBe(node.id)
    expect(adv.currentId).toBe(node.id)
    expect(node.text).toBe('You stand at a gate. What now?')
    expect(node.choices.map((c) => c.text)).toEqual(['Open the door', 'Run away', 'Call out'])
    expect(node.state).toEqual({ inventory: ['lamp'], stats: { Health: 10 }, flags: ['at the door'] })
    expect(node.rev).toBe(2)
    expect(done.text).toBe(node.text)

    // The phases, in order; the passage is saved before 'choosing' is announced.
    const phases = t.events.map((e) => e.status)
    expect(phases[0]).toBe('preparing')
    expect(phases.indexOf('waiting')).toBeLessThan(phases.indexOf('writing'))
    expect(phases.indexOf('writing')).toBeLessThan(phases.indexOf('choosing'))
    expect(phases[phases.length - 1]).toBe('done')
    const choosing = t.events.find((e) => e.status === 'choosing')
    expect(choosing?.resultNodeId).toBe(node.id)
    expect(t.events.find((e) => e.status === 'waiting')?.progress).toBeNull()

    // Two requests: a prose one and a structured one, as the design says.
    expect(t.requests).toHaveLength(2)
    const [passage, choosingReq] = t.requests
    expect(passage).toMatchObject({ prefitted: true, overflow: 'stop', disableThinking: true, adapters: null })
    expect(passage.sampling).toMatchObject({ jsonSchema: '', thinking: 'off', maxTokens: 256, seed: 7, temperature: DEFAULT_SAMPLING.temperature })
    expect(passage.messages.map((m) => m.role)).toEqual(['system', 'user'])
    expect(passage.messages[1].content).toBe('Begin the story.')
    expect(choosingReq.disableThinking).toBe(false)
    expect(choosingReq.sampling.jsonSchema).toContain('"setStats"')
    expect(choosingReq.sampling.thinking).toBe('auto')
    expect(choosingReq.messages.map((m) => m.role)).toEqual(['system', 'user', 'assistant', 'user'])
    expect(choosingReq.messages[2].content).toBe('You stand at a gate. What now?')
    expect(String(choosingReq.messages[3].content)).toContain('exactly 3 options')
    expect(String(choosingReq.messages[3].content)).toContain('starting stats')
    // The model gate is free again.
    expect(t.gate.heldBy()).toBeNull()
  })

  it('shows prompt-reading progress while waiting and streams the passage into the job', async () => {
    const t = await setup({ reply: (_r, _n, choosing) => (choosing ? CHOICES_JSON : { chunks: ['You ', 'stand ', 'there.'] }), throttleMs: 0 })
    await opening(t)
    expect(t.events.some((e) => e.status === 'waiting' && e.progress === 0.5)).toBe(true)
    const writing = t.events.filter((e) => e.status === 'writing').map((e) => e.text)
    expect(writing[writing.length - 1]).toBe('You stand there.')
    expect(writing[0]).toBe('You ')
  })

  it('refuses a second opening and an opening that names a parent', async () => {
    const t = await setup()
    await opening(t)
    await expect(t.runs.play({ adventureId: t.advId, parentId: null, input: { kind: 'start' } })).rejects.toThrow('already has its opening')
    await expect(t.runs.play({ adventureId: t.advId, parentId: 'x', input: { kind: 'start' } })).rejects.toThrow('not found')
    expect(t.gate.heldBy()).toBeNull()
  })
})

describe('playing on', () => {
  it('writes a child for a choice, with the story as alternating turns and the state in the last message', async () => {
    const t = await setup({ reply: (_r, n, choosing) => (choosing ? CHOICES_JSON : n === 0 ? 'The gate.' : 'The door creaks.') })
    const first = await opening(t)
    const root = (await t.store.get(t.advId)).nodes[first.resultNodeId as string]
    const job = await t.runs.play({ adventureId: t.advId, parentId: root.id, input: { kind: 'choice', choiceId: root.choices[0].id, text: 'IGNORED TEXT' } })
    expect(job.nodeId).toBe(root.id)
    expect(job.input).toEqual({ kind: 'choice', choiceId: root.choices[0].id, text: 'Open the door' })
    const done = await finished(t.runs, t.advId, job.id)
    expect(done.status).toBe('done')
    const req = t.requests[2]
    expect(req.messages.map((m) => m.role)).toEqual(['system', 'user', 'assistant', 'user'])
    expect(req.messages[2].content).toBe('The gate.')
    const last = String(req.messages[3].content)
    expect(last).toContain('The reader chooses: Open the door')
    expect(last).toContain('Inventory: lamp')
    expect(last).toContain('Health 10')
    expect(last).toContain('at the door')
    const adv = await t.store.get(t.advId)
    const child = adv.nodes[done.resultNodeId as string]
    expect(child.parentId).toBe(root.id)
    expect(child.input).toEqual({ kind: 'choice', choiceId: root.choices[0].id, text: 'Open the door' })
    expect(child.state.inventory).toEqual(['lamp'])
    expect(adv.currentId).toBe(child.id)
  })

  it('checks the input: a choice must be on the parent now, an ended parent takes only an action', async () => {
    const t = await setup()
    const first = await opening(t)
    const rootId = first.resultNodeId as string
    await expect(t.runs.play({ adventureId: t.advId, parentId: rootId, input: { kind: 'choice', choiceId: 'old-choice', text: 'x' } })).rejects.toThrow('not offered here any more')
    await expect(t.runs.play({ adventureId: t.advId, parentId: rootId, input: { kind: 'action', text: '   ' } })).rejects.toThrow('Type what you do')
    await expect(t.runs.play({ adventureId: t.advId, parentId: rootId, input: { kind: 'start' } })).rejects.toThrow('only be written first')
    await expect(t.runs.play({ adventureId: t.advId, parentId: 'nope', input: { kind: 'action', text: 'x' } })).rejects.toThrow('not found')
    await expect(t.runs.play({ adventureId: 'not an id', parentId: null, input: { kind: 'start' } })).rejects.toThrow('not an adventure id')
    // End the story at the root: choices are refused, an action is not.
    await t.store.setChoosing(t.advId, rootId, { choices: [], ended: true, changes: noChanges() })
    const choiceId = 'whatever'
    await expect(t.runs.play({ adventureId: t.advId, parentId: rootId, input: { kind: 'choice', choiceId, text: 'x' } })).rejects.toThrow('story ended')
    const job = await t.runs.play({ adventureId: t.advId, parentId: rootId, input: { kind: 'action', text: 'Wake up' } })
    expect((await finished(t.runs, t.advId, job.id)).status).toBe('done')
    expect(String(t.requests[t.requests.length - 2].messages.slice(-1)[0].content)).toContain('The reader does: Wake up')
    expect(t.gate.heldBy()).toBeNull()
  })
})

describe('stop and failure', () => {
  it('stopping while the passage is written saves nothing', async () => {
    let release: () => void = () => undefined
    const hold = new Promise<void>((r) => (release = r))
    const t = await setup({ reply: () => ({ chunks: ['Half a pass'], hold }) })
    const job = await t.runs.play({ adventureId: t.advId, parentId: null, input: { kind: 'start' } })
    await waitFor(() => t.events.some((e) => e.status === 'writing'))
    t.runs.abort(job.id)
    const done = await finished(t.runs, t.advId, job.id)
    release()
    expect(done.status).toBe('stopped')
    expect(done.resultNodeId).toBeNull()
    expect(done.text).toBe('')
    const adv = await t.store.get(t.advId)
    expect(Object.keys(adv.nodes)).toHaveLength(0)
    expect(adv.rootId).toBeNull()
    expect(t.gate.heldBy()).toBeNull()
  })

  it('a failure while the passage is written saves nothing', async () => {
    const t = await setup({
      reply: () => {
        throw new Error('server fell over')
      }
    })
    const job = await t.runs.play({ adventureId: t.advId, parentId: null, input: { kind: 'start' } })
    const done = await finished(t.runs, t.advId, job.id)
    expect(done.status).toBe('error')
    expect(done.error).toContain('server fell over')
    expect(Object.keys((await t.store.get(t.advId)).nodes)).toHaveLength(0)
    expect(t.gate.heldBy()).toBeNull()
  })

  it('stopping while the choices are written keeps the passage, with no choices', async () => {
    let release: () => void = () => undefined
    const hold = new Promise<void>((r) => (release = r))
    const t = await setup({ reply: (_r, _n, choosing) => (choosing ? { chunks: ['{"cho'], hold } : 'The gate.') })
    const job = await t.runs.play({ adventureId: t.advId, parentId: null, input: { kind: 'start' } })
    await waitFor(() => t.requests.length === 2)
    // The passage is on disk already, before the choices exist.
    const mid = await t.store.get(t.advId)
    expect(Object.values(mid.nodes).map((n) => n.text)).toEqual(['The gate.'])
    expect(Object.values(mid.nodes)[0].choices).toEqual([])
    expect(t.runs.list(t.advId)[0].status).toBe('choosing')
    t.runs.abort(job.id)
    const done = await finished(t.runs, t.advId, job.id)
    release()
    expect(done.status).toBe('stopped')
    expect(done.resultNodeId).not.toBeNull()
    expect(done.text).toBe('The gate.')
    const adv = await t.store.get(t.advId)
    const node = adv.nodes[done.resultNodeId as string]
    expect(node).toMatchObject({ text: 'The gate.', choices: [], ended: false, rev: 1 })
    expect(adv.currentId).toBe(node.id)
  })

  it('a failure while the choices are written keeps the passage and says why', async () => {
    const t = await setup({
      reply: (_r, _n, choosing) => {
        if (choosing) throw new Error('out of memory')
        return 'The gate.'
      }
    })
    const job = await t.runs.play({ adventureId: t.advId, parentId: null, input: { kind: 'start' } })
    const done = await finished(t.runs, t.advId, job.id)
    expect(done.status).toBe('error')
    expect(done.error).toContain('out of memory')
    const adv = await t.store.get(t.advId)
    expect(adv.nodes[done.resultNodeId as string].text).toBe('The gate.')
    expect(t.gate.heldBy()).toBeNull()
  })

  it('a hand edit made while the choices are written is kept: the late choices are not saved, the job ends done with a warning', async () => {
    let release: () => void = () => undefined
    const hold = new Promise<void>((r) => (release = r))
    const t = await setup({ reply: (_r, _n, choosing) => (choosing ? { chunks: [CHOICES_JSON], hold } : 'The gate.') })
    const job = await t.runs.play({ adventureId: t.advId, parentId: null, input: { kind: 'start' } })
    await waitFor(() => t.requests.length === 2)
    const nodes = (await t.store.get(t.advId)).nodes
    const saved = nodes[Object.keys(nodes)[0]]
    // The first save set rev 1; the reader edits the passage while the model is still answering.
    expect(saved.rev).toBe(1)
    await t.store.edit({ adventureId: t.advId, nodeId: saved.id, rev: saved.rev, text: 'Edited by hand.' })
    release()
    const done = await finished(t.runs, t.advId, job.id)
    expect(done.status).toBe('done')
    expect(done.warnings).toContain('The passage was edited while its choices were written, so they were not saved. Use New choices.')
    const node = (await t.store.get(t.advId)).nodes[saved.id]
    expect(node).toMatchObject({ text: 'Edited by hand.', choices: [], rev: 2, edited: true })
    expect(node.state.inventory).toEqual([])
    expect(t.gate.heldBy()).toBeNull()
  })

  it('a choosing request that cannot hold the newest passage fails plainly instead of going out without it', async () => {
    const t = await setup({ contextLength: 1800 })
    const first = await opening(t)
    expect(first.status).toBe('done')
    // Make the saved passage far bigger than the context can hold, then ask for new choices.
    const node = (await t.store.get(t.advId)).nodes[first.resultNodeId as string]
    await t.store.edit({ adventureId: t.advId, nodeId: node.id, rev: node.rev, text: 'long '.repeat(2500) })
    t.requests.length = 0
    const job = await t.runs.retry({ adventureId: t.advId, nodeId: node.id, what: 'choices' })
    const done = await finished(t.runs, t.advId, job.id)
    expect(done.status).toBe('error')
    expect(done.error).toContain('does not fit in the context')
    expect(t.requests).toHaveLength(0)
  })

  it('an unreadable choosing reply keeps the passage, ends done with a warning', async () => {
    const t = await setup({ reply: (_r, _n, choosing) => (choosing ? 'I would rather not say.' : 'The gate.') })
    const done = await opening(t)
    expect(done.status).toBe('done')
    expect(done.warnings.join(' ')).toContain('choices could not be read')
    const node = (await t.store.get(t.advId)).nodes[done.resultNodeId as string]
    expect(node.choices).toEqual([])
    expect(node.ended).toBe(false)
  })
})

describe('retries', () => {
  it('retryPassage writes a new sibling with the same input and makes it current; the old one stays', async () => {
    const t = await setup({ reply: (_r, n, choosing) => (choosing ? CHOICES_JSON : `Passage take ${n}.`) })
    const first = await opening(t)
    const rootId = first.resultNodeId as string
    const root = (await t.store.get(t.advId)).nodes[rootId]
    const step = await finished(t.runs, t.advId, (await t.runs.play({ adventureId: t.advId, parentId: rootId, input: { kind: 'choice', choiceId: root.choices[1].id, text: 'x' } })).id)
    const nodeId = step.resultNodeId as string
    const job = await t.runs.retry({ adventureId: t.advId, nodeId, what: 'passage' })
    expect(job.kind).toBe('retryPassage')
    expect(job.nodeId).toBe(nodeId)
    const done = await finished(t.runs, t.advId, job.id)
    expect(done.status).toBe('done')
    const adv = await t.store.get(t.advId)
    expect(done.resultNodeId).not.toBe(nodeId)
    const sibling = adv.nodes[done.resultNodeId as string]
    expect(sibling.parentId).toBe(rootId)
    expect(sibling.input).toEqual({ kind: 'choice', choiceId: root.choices[1].id, text: 'Run away' })
    expect(sibling.text).not.toBe(adv.nodes[nodeId].text)
    expect(adv.nodes[nodeId]).toBeDefined()
    expect(adv.currentId).toBe(sibling.id)
    // The request was built from the same parent: its last user message is the same step.
    expect(String(t.requests[t.requests.length - 2].messages.slice(-1)[0].content)).toContain('The reader chooses: Run away')
  })

  it('refuses to write the opening again', async () => {
    const t = await setup()
    const first = await opening(t)
    await expect(t.runs.retry({ adventureId: t.advId, nodeId: first.resultNodeId as string, what: 'passage' })).rejects.toThrow('opening cannot be written again')
    await expect(t.runs.retry({ adventureId: t.advId, nodeId: 'nope', what: 'choices' })).rejects.toThrow('not found')
    await expect(t.runs.retry({ adventureId: t.advId, nodeId: first.resultNodeId as string, what: 'oops' as never })).rejects.toThrow('not known')
    expect(t.gate.heldBy()).toBeNull()
  })

  it('retryChoices replaces choices, ended and changes in place and carries the state down the branch', async () => {
    let round = 0
    const t = await setup({
      reply: (_r, _n, choosing) => {
        if (!choosing) return 'Prose.'
        round++
        return round >= 3
          ? JSON.stringify({ choices: ['New one', 'New two'], ended: false, changes: { addItems: ['sword'], removeItems: ['lamp'], setStats: [], removeStats: [], addFlags: [], removeFlags: [] } })
          : CHOICES_JSON
      }
    })
    const first = await opening(t)
    const rootId = first.resultNodeId as string
    const root = (await t.store.get(t.advId)).nodes[rootId]
    const child = await finished(t.runs, t.advId, (await t.runs.play({ adventureId: t.advId, parentId: rootId, input: { kind: 'choice', choiceId: root.choices[0].id, text: '' } })).id)
    const before = await t.store.get(t.advId)
    const childBefore = before.nodes[child.resultNodeId as string]
    const requestsBefore = t.requests.length
    const job = await t.runs.retry({ adventureId: t.advId, nodeId: rootId, what: 'choices' })
    expect(job.kind).toBe('retryChoices')
    const done = await finished(t.runs, t.advId, job.id)
    expect(done.status).toBe('done')
    expect(done.resultNodeId).toBe(rootId)
    expect(done.text).toBe('')
    // Only the choosing request was made, about the saved passage.
    expect(t.requests.length).toBe(requestsBefore + 1)
    const adv = await t.store.get(t.advId)
    expect(adv.nodes[rootId].choices.map((c) => c.text)).toEqual(['New one', 'New two'])
    expect(adv.nodes[rootId].choices[0].id).not.toBe(root.choices[0].id)
    expect(adv.nodes[rootId].state.inventory).toEqual(['sword'])
    expect(adv.nodes[rootId].rev).toBe(before.nodes[rootId].rev + 1)
    // The branch below was recomputed (its own changes applied again) and its rev went up.
    expect(adv.nodes[childBefore.id].state.inventory).toEqual(['sword', 'lamp'])
    expect(adv.nodes[childBefore.id].rev).toBe(childBefore.rev + 1)
    // The reopen position does not move.
    expect(adv.currentId).toBe(before.currentId)
  })
})

describe('the model gate and the busy reason', () => {
  it('refuses a second job while one runs, with the gate\'s sentence, and frees the model at the end', async () => {
    let release: () => void = () => undefined
    const hold = new Promise<void>((r) => (release = r))
    const t = await setup({ reply: (_r, _n, choosing) => (choosing ? CHOICES_JSON : { chunks: ['x'], hold }) })
    const job = await t.runs.play({ adventureId: t.advId, parentId: null, input: { kind: 'start' } })
    expect(t.gate.heldBy()).toBe(`adventure:${job.id}`)
    await expect(t.runs.play({ adventureId: t.advId, parentId: null, input: { kind: 'start' } })).rejects.toThrow('already writing an adventure')
    expect(t.runs.hasRunning(t.advId)).toBe(true)
    release()
    await finished(t.runs, t.advId, job.id)
    expect(t.gate.heldBy()).toBeNull()
    expect(t.runs.hasRunning(t.advId)).toBe(false)
  })

  it('refuses while the busy reason says so, without taking the gate', async () => {
    let why: string | null = 'LLM Suite is writing something else on the computer.'
    const t = await setup({ busy: () => why })
    await expect(t.runs.play({ adventureId: t.advId, parentId: null, input: { kind: 'start' } })).rejects.toThrow('writing something else on the computer. Try again when it has finished.')
    expect(t.gate.heldBy()).toBeNull()
    why = null
    expect((await opening(t)).status).toBe('done')
  })

  it('gives the gate back when the request is refused after it was taken', async () => {
    const t = await setup()
    await expect(t.runs.play({ adventureId: t.advId, parentId: 'nope', input: { kind: 'action', text: 'x' } })).rejects.toThrow('not found')
    expect(t.gate.heldBy()).toBeNull()
  })

  it('a dismissed running job keeps the model until it has ended', async () => {
    let release: () => void = () => undefined
    const hold = new Promise<void>((r) => (release = r))
    const t = await setup({ reply: () => ({ chunks: ['x'], hold }) })
    const job = await t.runs.play({ adventureId: t.advId, parentId: null, input: { kind: 'start' } })
    await waitFor(() => t.events.some((e) => e.status === 'writing'))
    t.runs.dismiss(job.id)
    expect(t.runs.list(t.advId)).toEqual([])
    expect(typeof t.gate.take('adventure:other')).toBe('string')
    release()
    await waitFor(() => t.gate.heldBy() === null)
  })

  it('branchInUse is true for the parent of a running play and false afterwards', async () => {
    let release: () => void = () => undefined
    const hold = new Promise<void>((r) => (release = r))
    let slow = false
    const t = await setup({ reply: (_r, _n, choosing) => (choosing ? CHOICES_JSON : slow ? { chunks: ['x'], hold } : 'The gate.') })
    const first = await opening(t)
    const rootId = first.resultNodeId as string
    const root = (await t.store.get(t.advId)).nodes[rootId]
    slow = true
    const job = await t.runs.play({ adventureId: t.advId, parentId: rootId, input: { kind: 'choice', choiceId: root.choices[0].id, text: '' } })
    expect(t.runs.branchInUse(t.advId, new Set([rootId]))).toBe(true)
    expect(t.runs.branchInUse(t.advId, new Set(['other']))).toBe(false)
    expect(t.runs.branchInUse('different-adventure', new Set([rootId]))).toBe(false)
    release()
    await finished(t.runs, t.advId, job.id)
    expect(t.runs.branchInUse(t.advId, new Set([rootId]))).toBe(false)
  })

  it('WritingRuns takes the same gate for its jobs, refuses with its sentence and releases at the end', async () => {
    const gate = new ModelGate()
    const store: WritingStore = {
      hold: async () => {
        throw new Error('nope')
      },
      release: async () => undefined,
      projectState: async () => {
        throw new Error('unused')
      },
      readFile: async () => {
        throw new Error('unused')
      },
      statePatch: async () => {
        throw new Error('unused')
      }
    }
    const chat: WritingChat = { countText: async () => 1, countTokens: async () => 1, start: async () => undefined, abort: () => undefined, subscribe: () => () => undefined }
    const models: WritingModels = { list: () => [], ready: () => undefined, ensure: async () => Promise.reject(new Error('unused')) }
    const jobs: string[] = []
    const runs = new WritingRuns({ store, chat, models, gate, emit: (j) => jobs.push(j.status), holds: () => true, newId: () => 'wjob12345' })
    const req = { folder: 'C:\\novels\\book', action: 'continue' as const, sceneId: 's1', text: 'x', baseHash: 'h', cursor: 1, selection: null, note: '' }
    const held = gate.take('adventure:abc') as () => void
    expect(() => runs.run(req, 'phone:1')).toThrow('LLM Suite is already writing an adventure on the computer. Try again when it has finished.')
    held()
    const job = runs.run(req, 'phone:1')
    expect(gate.heldBy()).toBe(`writing:${job.id}`)
    expect(typeof gate.take('adventure:abc')).toBe('string')
    // The store refuses to open the project, so the job fails fast; the gate is free after that.
    await waitFor(() => jobs.includes('error'))
    expect(gate.heldBy()).toBeNull()
    expect(typeof gate.take('adventure:abc')).toBe('function')
  })
})

describe('events and bookkeeping', () => {
  it('throttles the streaming emits: a status change goes out at once, text at most once per gap', async () => {
    const chunks = Array.from({ length: 40 }, (_, i) => `w${i} `)
    const t = await setup({ reply: (_r, _n, choosing) => (choosing ? CHOICES_JSON : { chunks }), throttleMs: 60_000 })
    await opening(t)
    // Fake clock behaviour is not needed: with a huge gap only the status changes and one pending text emit exist.
    const writing = t.events.filter((e) => e.status === 'writing')
    expect(writing.length).toBeLessThanOrEqual(2)
    expect(t.events.length).toBeLessThan(12)
    expect(t.events[t.events.length - 1].status).toBe('done')
    expect(t.events[t.events.length - 1].text).toBe(chunks.join('').trim())
  })

  it('keeps jobs until dismissed and at most the limit of finished ones per adventure', async () => {
    const t = await setup({ maxFinished: 3 })
    const first = await opening(t)
    const rootId = first.resultNodeId as string
    const ids = [first.id]
    for (let i = 0; i < 5; i++) {
      const job = await t.runs.retry({ adventureId: t.advId, nodeId: rootId, what: 'choices' })
      ids.push(job.id)
      await finished(t.runs, t.advId, job.id)
    }
    const kept = t.runs.list(t.advId).map((j) => j.id)
    expect(kept).toEqual(ids.slice(-3))
    t.runs.dismiss(ids[ids.length - 1])
    expect(t.runs.list(t.advId).map((j) => j.id)).toEqual(ids.slice(-3, -1))
    t.runs.dismiss('not-a-job')
    t.runs.abort('not-a-job')
  })

  it('lists the model library with what is loaded', async () => {
    const t = await setup()
    expect(t.runs.models()).toEqual([
      { id: 'm1', name: 'Model One', loaded: true },
      { id: 'm2', name: 'Model Two', loaded: false }
    ])
  })
})

describe('the model and sampling', () => {
  it('uses the adventure\'s temperature over the default sampling', async () => {
    const t = await setup({ setup: { temperature: 1.3 } })
    await opening(t)
    expect(t.requests[0].sampling.temperature).toBe(1.3)
    expect(t.requests[0].sampling.topP).toBe(DEFAULT_SAMPLING.topP)
  })

  it('loads the setup\'s model when it is not loaded, and fails plainly when it cannot', async () => {
    const ok = await setup({ setup: { modelId: 'm2' } })
    expect((await opening(ok)).status).toBe('done')
    expect(ok.ensured).toEqual(['m2'])
    expect(ok.requests[0].modelId).toBe('m2')
    const bad = await setup({ setup: { modelId: 'm3' } })
    const job = await bad.runs.play({ adventureId: bad.advId, parentId: null, input: { kind: 'start' } })
    const done = await finished(bad.runs, bad.advId, job.id)
    expect(done.status).toBe('error')
    expect(done.error).toContain('could not be loaded')
    expect(bad.requests).toHaveLength(0)
    expect(bad.gate.heldBy()).toBeNull()
  })

  it('says plainly when no model is loaded', async () => {
    const t = await setup()
    t.models.ready = () => undefined
    const job = await t.runs.play({ adventureId: t.advId, parentId: null, input: { kind: 'start' } })
    const done = await finished(t.runs, t.advId, job.id)
    expect(done.status).toBe('error')
    expect(done.error).toContain('No model is loaded')
  })

  it('falls back to 8192 tokens of context when the model does not say', async () => {
    const t = await setup({ contextLength: null })
    expect((await opening(t)).status).toBe('done')
  })
})

describe('fitting a long story', () => {
  it('leaves older passages out in blocks, puts the left-out line in, and warns', async () => {
    const t = await setup({ contextLength: 2048, reply: (_r, _n, choosing) => (choosing ? CHOICES_JSON : 'Another passage. '.repeat(25)) })
    const first = await opening(t)
    let parent = first.resultNodeId as string
    for (let i = 0; i < 24; i++) {
      const node = (await t.store.addNode(t.advId, { parentId: parent, input: { kind: 'action', text: `step ${i}` }, text: `Passage ${i}. ${'filler '.repeat(60)}` })).node
      parent = node.id
    }
    t.requests.length = 0
    const job = await t.runs.play({ adventureId: t.advId, parentId: parent, input: { kind: 'action', text: 'carry on' } })
    const done = await finished(t.runs, t.advId, job.id)
    expect(done.error).toBeNull()
    expect(done.status).toBe('done')
    expect(done.warnings).toContain('Older passages were left out to fit the context.')
    const msgs = t.requests[0].messages
    expect(msgs.filter((m) => String(m.content).includes(LEFT_OUT_LINE))).toHaveLength(1)
    expect(String(msgs[0].content)).toContain('narrator')
    expect(String(msgs[msgs.length - 1].content)).toContain('The reader does: carry on')
    expect(String(msgs[2].content)).toContain('Another passage.')
    const body = msgs.slice(1, -1).filter((m) => m.role === 'assistant').length
    expect(body).toBeLessThan(26)
    expect((25 - body) % 8).toBe(0)
  })

  it('fails with a plain message when even the smallest request does not fit', async () => {
    const t = await setup({ contextLength: 600 })
    const job = await t.runs.play({ adventureId: t.advId, parentId: null, input: { kind: 'start' } })
    const done = await finished(t.runs, t.advId, job.id)
    expect(done.status).toBe('error')
    expect(done.error).toContain('does not fit in the context')
    expect(t.requests).toHaveLength(0)
  })
})
