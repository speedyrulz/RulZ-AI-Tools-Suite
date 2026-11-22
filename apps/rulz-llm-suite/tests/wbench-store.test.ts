import { afterEach, describe, expect, it } from 'vitest'
import { promises as fs } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { DEFAULT_SAMPLING } from '@shared/defaults'
import { BUILTIN_PROMPTS, newRun } from '@shared/writingBench'
import { WritingBenchStore } from '../src/main/wbench/store'

const dirs: string[] = []
async function setup() {
  const dir = await fs.mkdtemp(join(tmpdir(), 'wbench-')); dirs.push(dir)
  return { dir, store: new WritingBenchStore(dir), root: join(dir, 'writing-bench') }
}
function run() {
  return newRun({ label: 'test', prompts: BUILTIN_PROMPTS.slice(0, 1), candidates: [{ id: 'a', label: 'A', modelId: 'a', modelName: 'A', adapters: [], sampling: DEFAULT_SAMPLING, load: null }], seeds: [1], maxTokens: 600 }, 'now', 'run')
}
afterEach(async () => { await Promise.all(dirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true }))) })
describe('writing benchmark store', () => {
  it('saves, lists and deletes runs', async () => {
    const { store } = await setup(); const item = run()
    expect(await store.save(item)).toEqual(item); expect(await store.list()).toEqual([{ ok: true, run: item }])
    await store.delete(item.id); expect(await store.list()).toEqual([])
    await expect(store.delete('../bad')).rejects.toThrow()
  })
  it('lists damaged files without changing them and permits deletion', async () => {
    const { store, root } = await setup(); const dir = join(root, 'runs'); await fs.mkdir(dir, { recursive: true })
    const path = join(dir, 'broken.json'); await fs.writeFile(path, '{bad')
    expect(await store.list()).toMatchObject([{ ok: false, id: 'broken', file: path }])
    expect(await fs.readFile(path, 'utf8')).toBe('{bad'); await store.delete('broken')
    expect(await store.list()).toEqual([])
  })
  it('settles running on read without writing', async () => {
    const { store, root } = await setup(); const item = { ...run(), status: 'running' as const }
    await store.save(item); const path = join(root, 'runs', 'run.json'); const before = await fs.readFile(path)
    expect(await store.list()).toMatchObject([{ ok: true, run: { status: 'stopped' } }])
    expect(await fs.readFile(path)).toEqual(before)
  })
  it('round trips prompts and preserves unreadable prompts until save', async () => {
    const { store, root } = await setup(); const prompt = { ...BUILTIN_PROMPTS[0], id: 'mine', builtin: false }
    expect(await store.savePrompts([prompt])).toEqual([prompt]); expect(await store.prompts()).toEqual([prompt])
    const path = join(root, 'prompts.json'); await fs.writeFile(path, '{bad')
    expect(await store.prompts()).toEqual([]); expect(await fs.readFile(path, 'utf8')).toBe('{bad')
    await store.savePrompts([prompt]); expect(await store.prompts()).toEqual([prompt])
    expect((await fs.readdir(root)).some((name) => name.startsWith('prompts.json.corrupt-'))).toBe(true)
  })
})
