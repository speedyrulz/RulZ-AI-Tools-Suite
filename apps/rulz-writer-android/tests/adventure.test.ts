import { beforeEach, describe, expect, it, vi } from 'vitest'
import { emptyState, noChanges, type Adventure, type AdventureJob, type AdventureNode } from '../src/shared/adventure'
import { link } from '../src/lib/remote'
import { useAdventure } from '../src/stores/adventure'
import { choicesForEdit, parseStatValue } from '../src/components/adventure/editHelpers'
import { toast } from 'sonner'

const node = (id: string): AdventureNode => ({ id, parentId: null, input: { kind: 'start' }, text: id, choices: [], ended: false, changes: noChanges(), state: emptyState(), edited: false, rev: 1, createdAt: 1 })
const adventure = (): Adventure => ({ format: 1, id: 'a', setup: { title: 'A', genre: '', premise: '', character: '', tone: '', length: 'medium', choiceCount: 3, modelId: null, temperature: null }, nodes: { one: node('one') }, rootId: 'one', currentId: 'one', rev: 1, createdAt: 1, updatedAt: 1 })
const job = (id: string, resultNodeId: string | null): AdventureJob => ({ id, adventureId: 'a', kind: 'play', nodeId: 'one', input: { kind: 'action', text: 'go' }, status: 'choosing', progress: null, text: '', resultNodeId, error: null, warnings: [], startedAt: 1, endedAt: null })

describe('adventure store', () => {
  beforeEach(() => { vi.restoreAllMocks(); useAdventure.getState().close(); useAdventure.setState({ adventure: adventure(), position: 'one', list: [], jobs: {}, ownJobs: [], followedJobs: [], otherNode: null, starting: false }) })
  it('patches changes without moving this phone and ignores older changes', () => {
    useAdventure.getState().change({ id: 'a', rev: 2, nodes: [node('two')], currentId: 'two' })
    useAdventure.getState().change({ id: 'a', rev: 1, nodes: [node('old')] })
    expect(useAdventure.getState().position).toBe('one')
    expect(Object.keys(useAdventure.getState().adventure!.nodes)).toEqual(['one', 'two'])
  })
  it('follows its own result once, but offers another device result', async () => {
    vi.spyOn(link, 'invoke').mockResolvedValue(job('mine', null) as never)
    await useAdventure.getState().play({ adventureId: 'a', parentId: 'one', input: { kind: 'action', text: 'go' } })
    useAdventure.getState().job(job('mine', 'two'))
    expect(useAdventure.getState().position).toBe('one')
    useAdventure.getState().change({ id: 'a', rev: 2, nodes: [node('two')] })
    expect(useAdventure.getState().position).toBe('two')
    useAdventure.setState({ position: 'one' })
    useAdventure.getState().job(job('mine', 'two'))
    expect(useAdventure.getState().position).toBe('one')
    useAdventure.getState().job(job('theirs', 'three'))
    expect(useAdventure.getState().otherNode).toBe('three')
  })
  it('refreshes the open story and recovers a completed own job after reconnect', async () => {
    useAdventure.setState({ ownJobs: ['mine'] })
    vi.spyOn(link, 'invoke').mockImplementation(((channel: string) => Promise.resolve(channel === 'adventure:get' ? { ...adventure(), rev: 2, nodes: { one: node('one'), two: node('two') } } : [job('mine', 'two')])) as typeof link.invoke)
    await useAdventure.getState().refresh()
    expect(useAdventure.getState().adventure?.rev).toBe(2)
    expect(useAdventure.getState().position).toBe('two')
  })
  it('closes an adventure deleted on the computer', () => {
    useAdventure.getState().change({ id: 'a', rev: null })
    expect(useAdventure.getState().adventure).toBeNull()
  })
  it('buffers events during open and ignores an older open response', async () => {
    let resolveA!: (value: Adventure) => void
    const waiting = new Promise<Adventure>((resolve) => { resolveA = resolve })
    vi.spyOn(link, 'invoke').mockImplementation(((channel: string, id: string) => {
      if (channel === 'adventure:jobs') return Promise.resolve([])
      return id === 'a' ? waiting : Promise.resolve({ ...adventure(), id: 'b' })
    }) as typeof link.invoke)
    const first = useAdventure.getState().open('a')
    useAdventure.getState().change({ id: 'a', rev: 2, nodes: [node('two')] })
    useAdventure.getState().job(job('buffered', null))
    resolveA(adventure())
    expect(await first).toBe(true)
    expect(useAdventure.getState().adventure?.nodes.two).toBeDefined()
    expect(useAdventure.getState().jobs.buffered).toBeDefined()
    let resolveOld!: (value: Adventure) => void
    const old = new Promise<Adventure>((resolve) => { resolveOld = resolve })
    vi.mocked(link.invoke).mockImplementation(((channel: string, id: string) =>
      channel === 'adventure:jobs' ? Promise.resolve([]) : id === 'a' ? old : Promise.resolve({ ...adventure(), id: 'b' })) as typeof link.invoke)
    const older = useAdventure.getState().open('a')
    expect(await useAdventure.getState().open('b')).toBe(true)
    resolveOld(adventure())
    expect(await older).toBe(false)
    expect(useAdventure.getState().adventure?.id).toBe('b')
  })
  it('repairs position and replaces missing jobs while keeping newer listed events', async () => {
    let resolveJobs!: (value: AdventureJob[]) => void
    const listing = new Promise<AdventureJob[]>((resolve) => { resolveJobs = resolve })
    useAdventure.setState({ position: 'gone', jobs: { missing: job('missing', null) } })
    vi.spyOn(link, 'invoke').mockImplementation(((channel: string) =>
      channel === 'adventure:get' ? Promise.resolve({ ...adventure(), currentId: 'gone' }) : listing) as typeof link.invoke)
    const refresh = useAdventure.getState().refresh()
    const latest = { ...job('listed', null), text: 'new' }
    useAdventure.getState().job(latest)
    resolveJobs([job('listed', null)])
    await refresh
    expect(useAdventure.getState().position).toBe('one')
    expect(useAdventure.getState().jobs).toEqual({ listed: latest })
  })
  it('reports and dismisses a terminal own job once, including events before the call returns', async () => {
    let resolvePlay!: (value: AdventureJob) => void
    const waiting = new Promise<AdventureJob>((resolve) => { resolvePlay = resolve })
    const invoke = vi.spyOn(link, 'invoke').mockImplementation(((channel: string) =>
      channel === 'adventure:play' ? waiting : Promise.resolve()) as typeof link.invoke)
    const error = vi.spyOn(toast, 'error').mockImplementation(() => '' as never)
    const warning = vi.spyOn(toast, 'warning').mockImplementation(() => '' as never)
    const play = useAdventure.getState().play({ adventureId: 'a', parentId: 'one', input: { kind: 'action', text: 'go' } })
    expect(useAdventure.getState().starting).toBe(true)
    const finished = { ...job('terminal-test', null), status: 'error' as const, error: 'Failed', warnings: ['Warning'] }
    useAdventure.getState().job(finished)
    resolvePlay(job('terminal-test', null))
    await play
    useAdventure.getState().job(finished)
    expect(useAdventure.getState().starting).toBe(false)
    expect(error).toHaveBeenCalledTimes(1)
    expect(warning).toHaveBeenCalledTimes(1)
    expect(invoke).toHaveBeenCalledWith('adventure:dismiss', 'terminal-test')
  })
  it('closes after a not-found refresh and clears starting after a failed play', async () => {
    vi.spyOn(toast, 'error').mockImplementation(() => '' as never)
    vi.spyOn(toast, 'info').mockImplementation(() => '' as never)
    vi.spyOn(link, 'invoke').mockImplementation(((channel: string) => channel === 'adventure:jobs'
      ? Promise.resolve([]) : Promise.reject(new Error('That adventure was not found.'))) as typeof link.invoke)
    expect(await useAdventure.getState().play({ adventureId: 'a', parentId: 'one', input: { kind: 'action', text: 'go' } })).toBeNull()
    expect(useAdventure.getState().starting).toBe(false)
    await useAdventure.getState().refresh()
    expect(useAdventure.getState().adventure).toBeNull()
  })
})

describe('passage edit helpers', () => {
  it('parses finite numbers and preserves text and empty values', () => {
    expect(parseStatValue('  -12.5  ')).toBe(-12.5)
    expect(parseStatValue('0')).toBe(0)
    expect(parseStatValue('wary')).toBe('wary')
    expect(parseStatValue('')).toBe('')
    expect(parseStatValue('  ')).toBe('  ')
    expect(parseStatValue('Infinity')).toBe('Infinity')
    expect(parseStatValue('1e999')).toBe('1e999')
  })

  it('preserves each choice id after removal and reordering and leaves new rows without an id', () => {
    const rows = [
      { key: 'third', id: 'choice-c', text: ' Go west ' },
      { key: 'first', id: 'choice-a', text: 'Go east' },
      { key: 'new', text: 'Wait' },
      { key: 'empty', id: 'choice-b', text: '  ' }
    ]
    expect(choicesForEdit(rows)).toEqual([
      { id: 'choice-c', text: 'Go west' },
      { id: 'choice-a', text: 'Go east' },
      { text: 'Wait' }
    ])
  })
})
