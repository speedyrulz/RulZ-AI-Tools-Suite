import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { HISTORY_MAX_ENTRIES, type HistoryFile } from '@shared/history'
import type { LoraMergeJob } from '@shared/loraMerge'
import { DEFAULT_RECIPE, type MergeJob } from '@shared/merge'
import { HistoryStore, type StoredEntry } from '../src/main/history/store'

let dir: string
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'history-'))
})
afterEach(async () => {
  await rm(dir, { recursive: true, force: true })
})

const merge = (id: string, status: MergeJob['status'] = 'done', extra: Partial<MergeJob> = {}): StoredEntry => ({
  kind: 'merge',
  job: {
    id,
    recipe: { ...DEFAULT_RECIPE, a: 'a.safetensors', b: 'b.safetensors' },
    status,
    serverUrl: 'http://x',
    own: true,
    promptId: null,
    createdAt: 1,
    finishedAt: 2,
    progress: { node: 'X', value: 1, max: 2 },
    output: null,
    error: null,
    ...extra
  }
})
const lora = (id: string, status: LoraMergeJob['status'] = 'done'): StoredEntry => ({
  kind: 'loraMerge',
  job: { id, status, createdAt: 1, recipe: { kind: 'x' }, progress: { done: 1, total: 2 } } as unknown as LoraMergeJob
})

const store = (now = () => 1234): HistoryStore => new HistoryStore({ dir, now })
const readFileJson = async (): Promise<HistoryFile> => JSON.parse(await readFile(join(dir, 'history.json'), 'utf8')) as HistoryFile

describe('history store', () => {
  it('treats a missing file as empty history', async () => {
    expect(await store().list()).toEqual([])
  })

  it('records finished jobs newest first, clears progress, and survives a restart', async () => {
    const s = store()
    expect(await s.record(merge('m1'))).toBe(true)
    expect(await s.record(lora('l1', 'error'))).toBe(true)
    expect(await s.record(merge('m2', 'cancelled'))).toBe(true)
    const list = await s.list()
    expect(list.map((e) => `${e.kind}:${e.job.id}`)).toEqual(['merge:m2', 'loraMerge:l1', 'merge:m1'])
    expect(list.every((e) => e.job.progress === null)).toBe(true)
    const file = await readFileJson()
    expect(file.version).toBe(1)
    expect(file.entries).toHaveLength(3)
    expect((await store().list()).map((e) => e.job.id)).toEqual(['m2', 'l1', 'm1'])
    expect((await readdir(dir)).sort()).toEqual(['history.json'])
  })

  it('does not change the job it was given', async () => {
    const e = merge('m1')
    await store().record(e)
    expect((e.job as MergeJob).progress).not.toBeNull()
  })

  it('ignores jobs that are not final', async () => {
    const s = store()
    expect(await s.record(merge('a', 'running'))).toBe(false)
    expect(await s.record(merge('b', 'queued'))).toBe(false)
    expect(await s.record(merge('c', 'verifying'))).toBe(false)
    expect(await s.record(lora('d', 'saving'))).toBe(false)
    expect(await s.list()).toEqual([])
  })

  it('replaces an entry of the same kind and id in place, keeping other kinds with the same id', async () => {
    const s = store()
    await s.record(merge('x', 'error', { createdAt: 1 }))
    await s.record(lora('x'))
    await s.record(merge('y', 'done', { createdAt: 5 }))
    await s.record(merge('x', 'done', { createdAt: 1 }))
    const list = await s.list()
    expect(list.map((e) => `${e.kind}:${e.job.id}:${e.job.status}`)).toEqual(['merge:y:done', 'loraMerge:x:done', 'merge:x:done'])
  })

  it('orders new entries by createdAt, newest first', async () => {
    const s = store()
    await s.record(merge('mid', 'done', { createdAt: 20 }))
    await s.record(merge('old', 'done', { createdAt: 10 }))
    await s.record(merge('new', 'done', { createdAt: 30 }))
    expect((await s.list()).map((e) => e.job.id)).toEqual(['new', 'mid', 'old'])
  })

  it('updates a finished merge after Move into models: same place, no write when nothing changed', async () => {
    const s = store()
    const output = { movedTo: null } as unknown as MergeJob['output']
    await s.record(merge('old', 'done', { createdAt: 10, output }))
    await s.record(merge('new', 'done', { createdAt: 20 }))
    expect(await s.record(merge('old', 'done', { createdAt: 10, output }))).toBe(false)
    const moved = { movedTo: '/models/x.safetensors' } as unknown as MergeJob['output']
    expect(await s.record(merge('old', 'done', { createdAt: 10, output: moved }))).toBe(true)
    const list = await s.list()
    expect(list.map((e) => e.job.id)).toEqual(['new', 'old'])
    expect((list[1].job as MergeJob).output).toEqual(moved)
    expect((await readFileJson()).entries[1].job).toMatchObject({ output: moved })
  })

  it('keeps at most HISTORY_MAX_ENTRIES, dropping the oldest', async () => {
    const s = store()
    for (let i = 0; i < HISTORY_MAX_ENTRIES + 3; i++) await s.record(merge(`m${i}`))
    const list = await s.list()
    expect(list).toHaveLength(HISTORY_MAX_ENTRIES)
    expect(list[0].job.id).toBe(`m${HISTORY_MAX_ENTRIES + 2}`)
    expect(list.at(-1)?.job.id).toBe('m3')
  })

  it('removes an entry by kind and id and writes the file', async () => {
    const s = store()
    await s.record(merge('x'))
    await s.record(lora('x'))
    await s.remove('merge', 'x')
    expect((await s.list()).map((e) => e.kind)).toEqual(['loraMerge'])
    expect((await readFileJson()).entries).toHaveLength(1)
    await s.remove('merge', 'nope')
    expect(await s.list()).toHaveLength(1)
  })

  it('renames a corrupt file to history.json.bad-<timestamp> and starts empty', async () => {
    await writeFile(join(dir, 'history.json'), '{ not json')
    const s = store(() => 777)
    expect(await s.list()).toEqual([])
    expect(await readdir(dir)).toEqual(['history.json.bad-777'])
    expect(await readFile(join(dir, 'history.json.bad-777'), 'utf8')).toBe('{ not json')
    await s.record(merge('m1'))
    expect((await readdir(dir)).sort()).toEqual(['history.json', 'history.json.bad-777'])
    expect(await readFile(join(dir, 'history.json.bad-777'), 'utf8')).toBe('{ not json')
  })

  it('treats a file of the wrong shape as corrupt', async () => {
    await writeFile(join(dir, 'history.json'), JSON.stringify({ version: 2, entries: [] }))
    expect(await store(() => 5).list()).toEqual([])
    expect(await readdir(dir)).toEqual(['history.json.bad-5'])
  })

  it('drops malformed entries but keeps the good ones', async () => {
    const good = merge('ok')
    const noRecipe = merge('norecipe')
    delete (noRecipe.job as Partial<MergeJob>).recipe
    const badTime = merge('badtime', 'done', { createdAt: null as unknown as number })
    const noTime = merge('notime')
    delete (noTime.job as Partial<MergeJob>).createdAt
    await writeFile(
      join(dir, 'history.json'),
      JSON.stringify({ version: 1, entries: [good, { kind: 'grid', job: { id: 'g' } }, null, { kind: 'merge' }, noRecipe, badTime, noTime] })
    )
    const list = await store().list()
    expect(list.map((e) => e.job.id)).toEqual(['ok'])
  })

  it('does not treat a read error other than a missing file as empty history, and retries later', async () => {
    // A directory where the file should be: readFile fails with EISDIR, not ENOENT.
    await mkdir(join(dir, 'history.json'))
    const s = store()
    await expect(s.list()).rejects.toThrow(/could not be read/)
    await expect(s.record(merge('m1'))).rejects.toThrow(/could not be read/)
    await rm(join(dir, 'history.json'), { recursive: true })
    await writeFile(join(dir, 'history.json'), JSON.stringify({ version: 1, entries: [merge('kept')] }))
    expect((await s.list()).map((e) => e.job.id)).toEqual(['kept'])
    expect(await s.record(merge('m1'))).toBe(true)
    expect((await readFileJson()).entries.map((e) => e.job.id).sort()).toEqual(['kept', 'm1'])
  })

  it('refuses to write over a corrupt file it could not move aside', async () => {
    await writeFile(join(dir, 'history.json'), '{ not json')
    // The name the corrupt file would get is taken by a folder, so the rename fails.
    await mkdir(join(dir, 'history.json.bad-9'))
    await writeFile(join(dir, 'history.json.bad-9', 'x'), 'x')
    const s = store(() => 9)
    expect(await s.list()).toEqual([])
    await expect(s.record(merge('m1'))).rejects.toThrow(/could not be moved aside/)
    await expect(s.remove('merge', 'm1')).rejects.toThrow(/could not be moved aside/)
    expect(await readFile(join(dir, 'history.json'), 'utf8')).toBe('{ not json')
  })

  it('stores nothing for a new entry that the cap would cut at once', async () => {
    const entries = Array.from({ length: HISTORY_MAX_ENTRIES }, (_, i) => merge(`m${i}`, 'done', { createdAt: 10_000 - i }))
    await writeFile(join(dir, 'history.json'), JSON.stringify({ version: 1, entries }))
    const before = await readFile(join(dir, 'history.json'), 'utf8')
    const s = store()
    expect(await s.record(merge('ancient', 'done', { createdAt: 1 }))).toBe(false)
    expect(await readFile(join(dir, 'history.json'), 'utf8')).toBe(before)
    expect((await s.list()).some((e) => e.job.id === 'ancient')).toBe(false)
    // A newer one still goes in, and the oldest drops out.
    expect(await s.record(merge('fresh', 'done', { createdAt: 20_000 }))).toBe(true)
    const list = await s.list()
    expect(list).toHaveLength(HISTORY_MAX_ENTRIES)
    expect(list[0].job.id).toBe('fresh')
    expect(list.at(-1)?.job.id).toBe(`m${HISTORY_MAX_ENTRIES - 2}`)
  })

  it('keeps both entries when two jobs finish at once', async () => {
    const s = store()
    await Promise.all([s.record(merge('a')), s.record(lora('b')), s.record(merge('c'))])
    expect((await s.list()).map((e) => e.job.id).sort()).toEqual(['a', 'b', 'c'])
    expect((await readFileJson()).entries).toHaveLength(3)
    // A second store on the same folder sees them too.
    expect(await store().list()).toHaveLength(3)
  })
})
