/// <reference lib="dom" />
// Adventure mode, renderer side: the pure helpers and the store (position, jobs, events, conflicts) with a
// fake window API. No component is rendered.
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  emptyState,
  noChanges,
  type Adventure,
  type AdventureChange,
  type AdventureJob,
  type AdventureNode,
  type AdventureSetup
} from '../src/shared/adventure'
import {
  cleanSetup,
  draftOfState,
  emptySetup,
  isConflict,
  liveView,
  mapLabel,
  mapRows,
  moveItem,
  parseStat,
  setupProblem,
  shorten,
  siblingIndex,
  stateOfDraft,
  stepLine
} from '../src/renderer/src/lib/adventure'

const invokeMock = vi.fn(async (_channel: string, ..._args: unknown[]) => undefined as unknown)
const listeners: Record<string, (payload: never) => void> = {}
const toastMock = vi.hoisted(() => ({ error: vi.fn(), message: vi.fn(), success: vi.fn(), warning: vi.fn() }))
vi.mock('sonner', () => ({ toast: toastMock }))
vi.mock('../src/renderer/src/lib/api', () => ({
  invoke: (channel: string, ...args: unknown[]) => invokeMock(channel, ...args),
  on: (channel: string, fn: (payload: never) => void) => {
    listeners[channel] = fn
    return () => undefined
  },
  errorMessage: (e: unknown) => (e instanceof Error ? e.message : String(e))
}))

import { fixPosition, patchAdventure, startPosition, upsertMeta, useAdventure } from '../src/renderer/src/stores/adventure'

const SETUP: AdventureSetup = { ...emptySetup(), title: 'Tower', genre: 'Fantasy', premise: 'A tower.' }

function node(id: string, parentId: string | null, extra: Partial<AdventureNode> = {}): AdventureNode {
  return {
    id,
    parentId,
    input: parentId === null ? { kind: 'start' } : { kind: 'choice', choiceId: `c-${id}`, text: `Take ${id}` },
    text: `Text of ${id}`,
    choices: [],
    ended: false,
    changes: noChanges(),
    state: emptyState(),
    edited: false,
    rev: 1,
    createdAt: Number.parseInt(id.replace(/\D/g, '') || '0', 10),
    ...extra
  }
}

/** root -> a (choice c-a), b (choice c-b); a -> a1 (typed action). */
function fixture(): Adventure {
  const root = node('n0', null, { choices: [{ id: 'c-a', text: 'Go left' }, { id: 'c-b', text: 'Go right' }] })
  const a = node('n1', 'n0', { input: { kind: 'choice', choiceId: 'c-a', text: 'Go left' } })
  const b = node('n2', 'n0', { input: { kind: 'choice', choiceId: 'c-b', text: 'Go right' } })
  const a1 = node('n3', 'n1', { input: { kind: 'action', text: 'Wave' } })
  return {
    format: 1,
    id: 'adv-0001',
    setup: SETUP,
    nodes: { n0: root, n1: a, n2: b, n3: a1 },
    rootId: 'n0',
    currentId: 'n3',
    rev: 5,
    createdAt: 1,
    updatedAt: 10
  }
}

function job(extra: Partial<AdventureJob> = {}): AdventureJob {
  return {
    id: 'job-1',
    adventureId: 'adv-0001',
    kind: 'play',
    nodeId: 'n3',
    input: { kind: 'action', text: 'Wave' },
    status: 'writing',
    progress: null,
    text: '',
    resultNodeId: null,
    error: null,
    warnings: [],
    startedAt: 100,
    endedAt: null,
    ...extra
  }
}

const meta = (adv: Adventure) => ({ id: adv.id, title: adv.setup.title, genre: '', nodeCount: 4, depth: 3, ended: false, createdAt: 1, updatedAt: adv.updatedAt })

let jobCounter = 0
/** A job id never used before: the store remembers ids across tests. */
const freshId = (): string => `job-${++jobCounter}-${Math.random().toString(36).slice(2, 8)}`

function emit(channel: 'adventure:job' | 'adventure:changed', payload: AdventureJob | AdventureChange): void {
  listeners[channel](payload as never)
}

/** Opens `adv` in the store through adventure:get, with `jobs` as adventure:jobs. */
async function openFixture(adv = fixture(), jobs: AdventureJob[] = []): Promise<void> {
  invokeMock.mockImplementation(async (channel: string) => {
    if (channel === 'adventure:get') return adv
    if (channel === 'adventure:jobs') return jobs
    return undefined
  })
  await useAdventure.getState().open(adv.id)
}

beforeAll(async () => {
  await useAdventure.getState().init()
})

beforeEach(() => {
  invokeMock.mockReset()
  invokeMock.mockResolvedValue(undefined)
  for (const f of Object.values(toastMock)) f.mockClear()
  useAdventure.setState({ list: [], listLoaded: false, adv: null, opening: false, position: null, job: null, remoteNodeId: null, panelOpen: true, panelTab: 'state' })
})

describe('adventure helpers', () => {
  it('lists the tree as an indented outline with the path marked', () => {
    const rows = mapRows(fixture(), 'n3')
    expect(rows.map((r) => [r.node.id, r.depth, r.onPath, r.isPosition])).toEqual([
      ['n0', 0, true, false],
      ['n1', 1, true, false],
      ['n3', 2, true, true],
      ['n2', 1, false, false]
    ])
  })

  it('gives an empty outline before the opening exists', () => {
    expect(mapRows({ ...fixture(), nodes: {}, rootId: null }, null)).toEqual([])
  })

  it('shortens step text and names the opening', () => {
    expect(shorten('  a   b  ', 20)).toBe('a b')
    expect(shorten('x'.repeat(50), 10)).toHaveLength(10)
    expect(shorten('x'.repeat(50), 10).endsWith('…')).toBe(true)
    const adv = fixture()
    expect(mapLabel(adv.nodes.n0)).toBe('The opening')
    expect(mapLabel(adv.nodes.n3)).toBe('You: Wave')
    expect(stepLine(adv.nodes.n1.input)).toBe('You chose: Go left')
    expect(stepLine(adv.nodes.n0.input)).toBe('')
  })

  it('counts the other paths from the same step', () => {
    const adv = fixture()
    expect(siblingIndex(adv, 'n1')).toEqual({ index: 0, count: 2, prev: null, next: 'n2' })
    expect(siblingIndex(adv, 'n2')).toEqual({ index: 1, count: 2, prev: 'n1', next: null })
    expect(siblingIndex(adv, 'n0').count).toBe(1)
  })

  it('decides what the reading column shows while a job runs', () => {
    expect(liveView(null, 'n3').kind).toBe('none')
    expect(liveView(job({ status: 'done' }), 'n3').kind).toBe('none')
    expect(liveView(job(), 'n3').kind).toBe('passage')
    // The reader moved elsewhere while the passage is written: nothing is streamed into this column.
    expect(liveView(job(), 'n2').kind).toBe('none')
    const retry = liveView(job({ kind: 'retryPassage', nodeId: 'n3' }), 'n3')
    expect(retry.kind === 'passage' && retry.replaces).toBe('n3')
    expect(liveView(job({ status: 'choosing', resultNodeId: 'n4' }), 'n4').kind).toBe('choices')
    expect(liveView(job({ status: 'choosing', resultNodeId: 'n4' }), 'n3').kind).toBe('none')
    expect(liveView(job({ kind: 'retryChoices', nodeId: 'n3', status: 'choosing' }), 'n3').kind).toBe('choices')
  })

  it('reads the conflict refusal', () => {
    expect(isConflict('This passage was changed somewhere else. Reload it and try again.')).toBe(true)
    expect(isConflict('No model is loaded.')).toBe(false)
  })

  it('moves list items and keeps out-of-range moves as they are', () => {
    expect(moveItem(['a', 'b', 'c'], 0, 2)).toEqual(['b', 'c', 'a'])
    const same = ['a', 'b']
    expect(moveItem(same, 0, 5)).toBe(same)
  })

  it('turns the state form into a state and back', () => {
    const st = { inventory: ['rusty key'], stats: { Health: 8, Mood: 'wary' }, flags: ['met the ferryman'] }
    expect(stateOfDraft(draftOfState(st))).toEqual(st)
    expect(
      stateOfDraft({
        inventory: ['  lamp ', '', 'LAMP'],
        flags: [' '],
        stats: [
          { name: 'Gold', value: '12' },
          { name: ' ', value: '1' },
          { name: 'gold', value: '13' }
        ]
      })
    ).toEqual({ inventory: ['lamp'], flags: [], stats: { gold: 13 } })
    expect(parseStat(' 3.5 ')).toBe(3.5)
    expect(parseStat('wary')).toBe('wary')
    expect(parseStat('')).toBe('')
  })

  it('checks the new-adventure form', () => {
    const form = emptySetup()
    expect(setupProblem(form, false, null)).toBe('Give the adventure a title.')
    expect(setupProblem({ ...form, title: 'T' }, false, null)).toBe('Say what the story is about.')
    expect(setupProblem({ ...form, title: 'T', premise: 'P' }, false, null)).toBeNull()
    expect(setupProblem({ ...form, title: 'T' }, true, null)).toBe('Pick a Writing project.')
    // From a project the premise is optional.
    expect(setupProblem({ ...form, title: 'T' }, true, 'C:/p')).toBeNull()
    expect(cleanSetup({ ...form, title: ' T ', choiceCount: 9 })).toMatchObject({ title: 'T', choiceCount: 4 })
    expect(cleanSetup({ ...form, choiceCount: 1 }).choiceCount).toBe(2)
  })
})

describe('patching the held adventure', () => {
  it('applies changes and keeps a newer node when an older copy arrives late', () => {
    const adv = fixture()
    const newer = { ...adv.nodes.n1, text: 'Edited', rev: 3 }
    const patched = patchAdventure(adv, { id: adv.id, rev: 6, nodes: [newer] })
    expect(patched.nodes.n1.text).toBe('Edited')
    const late = patchAdventure(patched, { id: adv.id, rev: 5, nodes: [{ ...adv.nodes.n1, rev: 2 }], setup: { ...SETUP, title: 'Old' }, currentId: 'n0' })
    expect(late.nodes.n1.text).toBe('Edited')
    expect(late.setup.title).toBe('Tower')
    expect(late.currentId).toBe('n3')
  })

  it('ignores another adventure and a delete', () => {
    const adv = fixture()
    expect(patchAdventure(adv, { id: 'other', rev: 9 })).toBe(adv)
    expect(patchAdventure(adv, { id: adv.id, rev: null })).toBe(adv)
  })

  it('finds a position that exists', () => {
    const adv = fixture()
    expect(fixPosition(adv, 'n1')).toBe('n1')
    expect(fixPosition(adv, 'gone')).toBe('n3')
    expect(fixPosition({ ...adv, currentId: 'gone' }, 'gone')).toBe('n0')
    expect(startPosition({ ...adv, currentId: null })).toBe('n0')
    expect(startPosition({ ...adv, nodes: {}, rootId: null, currentId: null })).toBeNull()
  })

  it('keeps the list newest first', () => {
    const adv = fixture()
    const list = upsertMeta([{ ...meta(adv), id: 'a', updatedAt: 5 }, { ...meta(adv), id: 'b', updatedAt: 1 }], { ...meta(adv), id: 'b', updatedAt: 9 })
    expect(list.map((m) => m.id)).toEqual(['b', 'a'])
  })
})

describe('adventure store', () => {
  it('opens at the saved position and picks up a running job', async () => {
    const running = job({ id: freshId(), status: 'waiting' })
    await openFixture(fixture(), [job({ id: freshId(), status: 'done' }), running])
    const s = useAdventure.getState()
    expect(s.position).toBe('n3')
    expect(s.job?.id).toBe(running.id)
    expect(invokeMock).toHaveBeenCalledWith('adventure:jobs', 'adv-0001')
  })

  it('goes back here: moves only this window and saves the reopen position', async () => {
    await openFixture()
    invokeMock.mockResolvedValue({ id: 'adv-0001', rev: 6, currentId: 'n1', meta: meta(fixture()) } satisfies AdventureChange)
    useAdventure.getState().go('n1')
    expect(useAdventure.getState().position).toBe('n1')
    expect(invokeMock).toHaveBeenCalledWith('adventure:setCurrent', 'adv-0001', 'n1')
    await vi.waitFor(() => expect(useAdventure.getState().adv?.currentId).toBe('n1'))
    expect(useAdventure.getState().position).toBe('n1')
  })

  it('moves to a result this window started, once the node arrives', async () => {
    await openFixture()
    const id = freshId()
    invokeMock.mockImplementation(async (channel: string) => (channel === 'adventure:play' ? job({ id, status: 'preparing' }) : undefined))
    await useAdventure.getState().play({ kind: 'action', text: 'Wave' })
    expect(invokeMock).toHaveBeenCalledWith('adventure:play', { adventureId: 'adv-0001', parentId: 'n3', input: { kind: 'action', text: 'Wave' } })
    // The job event comes before the change that carries the new node.
    emit('adventure:job', job({ id, status: 'choosing', resultNodeId: 'n4', text: 'New' }))
    expect(useAdventure.getState().position).toBe('n3')
    emit('adventure:changed', { id: 'adv-0001', rev: 6, nodes: [node('n4', 'n3')], currentId: 'n4' })
    expect(useAdventure.getState().position).toBe('n4')
    expect(useAdventure.getState().remoteNodeId).toBeNull()
  })

  it('treats a job event that arrives before the play call returns as its own', async () => {
    await openFixture()
    const id = freshId()
    let release: (j: AdventureJob) => void = () => undefined
    invokeMock.mockImplementation((channel: string) => (channel === 'adventure:play' ? new Promise((res) => (release = res as (j: AdventureJob) => void)) : Promise.resolve(undefined)))
    const call = useAdventure.getState().play({ kind: 'action', text: 'Wave' })
    emit('adventure:changed', { id: 'adv-0001', rev: 6, nodes: [node('n4', 'n3')] })
    emit('adventure:job', job({ id, status: 'choosing', resultNodeId: 'n4' }))
    release(job({ id, status: 'preparing' }))
    await call
    expect(useAdventure.getState().position).toBe('n4')
    // The older copy the call returned did not replace the newer job state.
    expect(useAdventure.getState().job?.status).toBe('choosing')
    expect(useAdventure.getState().remoteNodeId).toBeNull()
  })

  it('does not move for a job started elsewhere, and offers a jump', async () => {
    await openFixture()
    emit('adventure:changed', { id: 'adv-0001', rev: 6, nodes: [node('n4', 'n3')], currentId: 'n4' })
    emit('adventure:job', job({ id: freshId(), status: 'choosing', resultNodeId: 'n4' }))
    expect(useAdventure.getState().position).toBe('n3')
    expect(useAdventure.getState().remoteNodeId).toBe('n4')
    invokeMock.mockResolvedValue({ id: 'adv-0001', rev: 7, currentId: 'n4' })
    useAdventure.getState().goRemote()
    expect(useAdventure.getState().position).toBe('n4')
    expect(useAdventure.getState().remoteNodeId).toBeNull()
  })

  it('reports a finished own job once and forgets it', async () => {
    await openFixture()
    const id = freshId()
    invokeMock.mockImplementation(async (channel: string) => (channel === 'adventure:play' ? job({ id, status: 'preparing' }) : undefined))
    await useAdventure.getState().play({ kind: 'action', text: 'Wave' })
    const failed = job({ id, status: 'error', error: 'The model ran out of memory.', warnings: ['Older passages were left out.'] })
    emit('adventure:job', failed)
    emit('adventure:job', failed)
    expect(toastMock.error).toHaveBeenCalledTimes(1)
    expect(toastMock.error).toHaveBeenCalledWith('The model ran out of memory.')
    expect(toastMock.warning).toHaveBeenCalledWith('Older passages were left out.')
    expect(invokeMock).toHaveBeenCalledWith('adventure:dismiss', id)
  })

  it('says why a job could not start', async () => {
    await openFixture()
    invokeMock.mockImplementation(async (channel: string) => {
      if (channel === 'adventure:play') throw new Error('Another model job is running.')
      return undefined
    })
    await useAdventure.getState().play({ kind: 'action', text: 'Wave' })
    expect(toastMock.error).toHaveBeenCalledWith('Another model job is running.')
    expect(useAdventure.getState().job).toBeNull()
  })

  it('does not start a second job while one runs', async () => {
    await openFixture(fixture(), [job({ id: freshId(), status: 'writing' })])
    invokeMock.mockClear()
    await useAdventure.getState().play({ kind: 'action', text: 'Wave' })
    await useAdventure.getState().retry('n3', 'passage')
    expect(invokeMock).not.toHaveBeenCalled()
  })

  it('goes to a choice already taken instead of writing it again, unless forced', async () => {
    await openFixture()
    useAdventure.setState({ position: 'n0' })
    invokeMock.mockResolvedValue({ id: 'adv-0001', rev: 6, currentId: 'n1' })
    await useAdventure.getState().choose('c-a', 'Go left')
    expect(useAdventure.getState().position).toBe('n1')
    expect(invokeMock).not.toHaveBeenCalledWith('adventure:play', expect.anything())

    useAdventure.setState({ position: 'n0' })
    invokeMock.mockImplementation(async (channel: string) => (channel === 'adventure:play' ? job({ id: freshId(), nodeId: 'n0', status: 'preparing' }) : undefined))
    await useAdventure.getState().choose('c-a', 'Go left', true)
    expect(invokeMock).toHaveBeenCalledWith('adventure:play', { adventureId: 'adv-0001', parentId: 'n0', input: { kind: 'choice', choiceId: 'c-a', text: 'Go left' } })

    // A choice never taken is written.
    invokeMock.mockClear()
    useAdventure.setState({ job: null })
    await useAdventure.getState().choose('c-new', 'Climb')
    expect(invokeMock).toHaveBeenCalledWith('adventure:play', { adventureId: 'adv-0001', parentId: 'n0', input: { kind: 'choice', choiceId: 'c-new', text: 'Climb' } })
  })

  it('writes the opening with parentId null', async () => {
    const empty: Adventure = { ...fixture(), nodes: {}, rootId: null, currentId: null }
    await openFixture(empty)
    expect(useAdventure.getState().position).toBeNull()
    invokeMock.mockImplementation(async (channel: string) => (channel === 'adventure:play' ? job({ id: freshId(), nodeId: null, status: 'preparing', input: { kind: 'start' } }) : undefined))
    await useAdventure.getState().play({ kind: 'start' })
    expect(invokeMock).toHaveBeenCalledWith('adventure:play', { adventureId: 'adv-0001', parentId: null, input: { kind: 'start' } })
  })

  it('creates with the world folder, opens the adventure and plays the opening', async () => {
    const created: Adventure = { ...fixture(), id: 'adv-0002', nodes: {}, rootId: null, currentId: null }
    invokeMock.mockImplementation(async (channel: string) => {
      if (channel === 'adventure:create') return created
      if (channel === 'adventure:play') return job({ id: freshId(), adventureId: 'adv-0002', nodeId: null, status: 'preparing', input: { kind: 'start' } })
      return undefined
    })
    const ok = await useAdventure.getState().startNew(SETUP, 'C:/books/tower')
    expect(ok).toBe(true)
    expect(invokeMock).toHaveBeenCalledWith('adventure:create', SETUP, 'C:/books/tower')
    expect(invokeMock).toHaveBeenCalledWith('adventure:play', { adventureId: 'adv-0002', parentId: null, input: { kind: 'start' } })
    expect(useAdventure.getState().adv?.id).toBe('adv-0002')
    expect(useAdventure.getState().list.map((m) => m.id)).toContain('adv-0002')
  })

  it('goes back to the list with a message when the open adventure was deleted elsewhere', async () => {
    await openFixture()
    emit('adventure:changed', { id: 'adv-0001', rev: null })
    const s = useAdventure.getState()
    expect(s.adv).toBeNull()
    expect(s.position).toBeNull()
    expect(toastMock.message).toHaveBeenCalledWith('This adventure was deleted on another device.')
  })

  it('moves the position when its branch was deleted', async () => {
    await openFixture()
    emit('adventure:changed', { id: 'adv-0001', rev: 6, removedIds: ['n1', 'n3'], currentId: 'n0' })
    expect(useAdventure.getState().adv?.nodes.n3).toBeUndefined()
    expect(useAdventure.getState().position).toBe('n0')
  })

  it('saves an edit with the node rev and patches from the returned change', async () => {
    await openFixture()
    invokeMock.mockResolvedValue({ id: 'adv-0001', rev: 6, nodes: [{ ...fixture().nodes.n1, text: 'Better', edited: true, rev: 2 }] } satisfies AdventureChange)
    const result = await useAdventure.getState().edit({ adventureId: 'adv-0001', nodeId: 'n1', rev: 1, text: 'Better' })
    expect(result).toBe('ok')
    expect(useAdventure.getState().adv?.nodes.n1.text).toBe('Better')
  })

  it('reloads the adventure and says so on an edit conflict', async () => {
    await openFixture()
    const fresh = fixture()
    fresh.nodes.n1 = { ...fresh.nodes.n1, text: 'Changed on the phone', rev: 4 }
    invokeMock.mockImplementation(async (channel: string) => {
      if (channel === 'adventure:edit') throw new Error('This passage was changed somewhere else. Reload it and try again.')
      if (channel === 'adventure:get') return fresh
      return undefined
    })
    const result = await useAdventure.getState().edit({ adventureId: 'adv-0001', nodeId: 'n1', rev: 1, text: 'Mine' })
    expect(result).toBe('conflict')
    expect(toastMock.error).toHaveBeenCalledWith(expect.stringContaining('changed somewhere else'))
    await vi.waitFor(() => expect(useAdventure.getState().adv?.nodes.n1.text).toBe('Changed on the phone'))
  })

  it('deletes a branch with the node rev and keeps a valid position', async () => {
    const adv = fixture()
    adv.nodes.n1 = { ...adv.nodes.n1, rev: 7 }
    await openFixture(adv)
    invokeMock.mockResolvedValue({ id: 'adv-0001', rev: 6, removedIds: ['n1', 'n3'], currentId: 'n0' } satisfies AdventureChange)
    const result = await useAdventure.getState().deleteBranch('n1')
    expect(result).toBe('ok')
    expect(invokeMock).toHaveBeenCalledWith('adventure:deleteBranch', 'adv-0001', 'n1', 7)
    expect(useAdventure.getState().position).toBe('n0')
  })

  it('exports the path to this window position and reports where it was saved', async () => {
    await openFixture()
    useAdventure.setState({ position: 'n1' })
    invokeMock.mockResolvedValue('C:/Stories/Tower.md')
    await useAdventure.getState().exportStory()
    expect(invokeMock).toHaveBeenCalledWith('adventure:saveExport', 'adv-0001', 'n1')
    expect(toastMock.success).toHaveBeenCalledWith('Saved to C:/Stories/Tower.md')
    toastMock.success.mockClear()
    invokeMock.mockResolvedValue(null)
    await useAdventure.getState().exportStory()
    expect(toastMock.success).not.toHaveBeenCalled()
  })

  it('stops the running job', async () => {
    const id = freshId()
    await openFixture(fixture(), [job({ id, status: 'writing' })])
    useAdventure.getState().abort()
    expect(invokeMock).toHaveBeenCalledWith('adventure:abort', id)
  })

  it('renames through updateSetup and updates the list row', async () => {
    useAdventure.setState({ list: [meta(fixture())] })
    invokeMock.mockResolvedValue({ id: 'adv-0001', rev: 6, setup: { ...SETUP, title: 'New name' }, meta: { ...meta(fixture()), title: 'New name', updatedAt: 50 } } satisfies AdventureChange)
    await useAdventure.getState().rename('adv-0001', 'New name')
    expect(invokeMock).toHaveBeenCalledWith('adventure:updateSetup', 'adv-0001', { title: 'New name' })
    expect(useAdventure.getState().list[0].title).toBe('New name')
  })

  it('removes an adventure from the list and closes it when open', async () => {
    await openFixture()
    useAdventure.setState({ list: [meta(fixture())] })
    await useAdventure.getState().remove('adv-0001')
    expect(invokeMock).toHaveBeenCalledWith('adventure:delete', 'adv-0001')
    expect(useAdventure.getState().adv).toBeNull()
    expect(useAdventure.getState().list).toEqual([])
  })

  it('ignores events of other adventures but keeps their list rows current', async () => {
    await openFixture()
    emit('adventure:job', job({ id: freshId(), adventureId: 'other-adv', status: 'writing' }))
    expect(useAdventure.getState().job).toBeNull()
    emit('adventure:changed', { id: 'other-adv', rev: 2, meta: { ...meta(fixture()), id: 'other-adv', updatedAt: 99 } })
    expect(useAdventure.getState().list.map((m) => m.id)).toEqual(['other-adv', 'adv-0001'])
  })
})

describe('review fixes', () => {
  it('never adds a node from a stale change, but still applies its removals', () => {
    const adv = fixture()
    const afterDelete = patchAdventure(adv, { id: adv.id, rev: 10, removedIds: ['n1', 'n3'], currentId: 'n0' })
    expect(afterDelete.nodes.n3).toBeUndefined()
    // The reply of an earlier call lands late, carrying a node of the deleted branch.
    const late = patchAdventure(afterDelete, { id: adv.id, rev: 9, nodes: [node('n3', 'n1', { rev: 2 })] })
    expect(late.nodes.n3).toBeUndefined()
    expect(late.nodes.n1).toBeUndefined()
    // A stale removal still applies.
    expect(patchAdventure(adv, { id: adv.id, rev: 3, removedIds: ['n2'] }).nodes.n2).toBeUndefined()
    // A current change still adds nodes.
    expect(patchAdventure(afterDelete, { id: adv.id, rev: 11, nodes: [node('n9', 'n0')] }).nodes.n9).toBeDefined()
  })

  it('a job from another device that wins the gate during our refused start call stays remote', async () => {
    await openFixture()
    const other = freshId()
    let fail: (e: Error) => void = () => undefined
    invokeMock.mockImplementation((channel: string) => (channel === 'adventure:play' ? new Promise((_res, rej) => (fail = rej)) : Promise.resolve(undefined)))
    const call = useAdventure.getState().play({ kind: 'action', text: 'Wave' })
    emit('adventure:changed', { id: 'adv-0001', rev: 6, nodes: [node('n4', 'n3')] })
    emit('adventure:job', job({ id: other, status: 'choosing', resultNodeId: 'n4' }))
    fail(new Error('Another model job is running.'))
    await call
    emit('adventure:job', job({ id: other, status: 'done', resultNodeId: 'n4' }))
    expect(useAdventure.getState().position).toBe('n3')
    expect(useAdventure.getState().remoteNodeId).toBe('n4')
    expect(invokeMock).not.toHaveBeenCalledWith('adventure:dismiss', other)
  })

  it('adopts the job the call names, even when its result was already shown as a remote note', async () => {
    await openFixture()
    const mine = freshId()
    const other = freshId()
    let release: (j: AdventureJob) => void = () => undefined
    invokeMock.mockImplementation((channel: string) => (channel === 'adventure:play' ? new Promise((res) => (release = res as (j: AdventureJob) => void)) : Promise.resolve(undefined)))
    const call = useAdventure.getState().play({ kind: 'action', text: 'Wave' })
    emit('adventure:changed', { id: 'adv-0001', rev: 6, nodes: [node('n4', 'n3')] })
    emit('adventure:job', job({ id: mine, status: 'choosing', resultNodeId: 'n4' }))
    emit('adventure:job', job({ id: other, status: 'writing', startedAt: 50 }))
    expect(useAdventure.getState().remoteNodeId).toBe('n4')
    release(job({ id: mine, status: 'preparing' }))
    await call
    expect(useAdventure.getState().position).toBe('n4')
    expect(useAdventure.getState().remoteNodeId).toBeNull()
  })

  it('replays events that arrived while the adventure was being opened', async () => {
    const snapshot = fixture()
    let release: () => void = () => undefined
    invokeMock.mockImplementation(async (channel: string) => {
      if (channel === 'adventure:get') {
        await new Promise<void>((res) => (release = res))
        return snapshot
      }
      return []
    })
    const opening = useAdventure.getState().open('adv-0001')
    await vi.waitFor(() => expect(invokeMock).toHaveBeenCalledWith('adventure:get', 'adv-0001'))
    const runner = freshId()
    emit('adventure:changed', { id: 'adv-0001', rev: 6, nodes: [node('n4', 'n3')], currentId: 'n4' })
    emit('adventure:job', job({ id: runner, status: 'choosing', resultNodeId: 'n4' }))
    release()
    await opening
    const s = useAdventure.getState()
    expect(s.adv?.nodes.n4).toBeDefined()
    expect(s.job?.id).toBe(runner)
    expect(s.position).toBe('n3')
    expect(s.remoteNodeId).toBe('n4')
  })
})
