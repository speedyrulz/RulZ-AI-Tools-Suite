import { promises as fs } from 'node:fs'
import { join } from 'node:path'
import { cleanPrompts, cleanRun, type BenchPrompt, type BenchRunEntry, type WritingBenchRun } from '@shared/writingBench'
import { readJson, writeJsonAtomic } from '../util/jsonFile'

export class WritingBenchStore {
  private readonly runs: string
  private readonly promptFile: string
  constructor(userData: string) {
    const root = join(userData, 'writing-bench')
    this.runs = join(root, 'runs')
    this.promptFile = join(root, 'prompts.json')
  }
  private path(id: string): string {
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(id)) throw new Error('Invalid benchmark run id.')
    return join(this.runs, `${id}.json`)
  }
  async list(): Promise<BenchRunEntry[]> {
    let files: string[]
    try { files = await fs.readdir(this.runs) } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') return []
      throw err
    }
    return Promise.all(files.filter((file) => file.endsWith('.json')).sort().map(async (file): Promise<BenchRunEntry> => {
      const id = file.slice(0, -5), path = join(this.runs, file)
      try {
        const parsed: unknown = JSON.parse((await fs.readFile(path, 'utf8')).replace(/^\uFEFF/, ''))
        const result = cleanRun(parsed)
        if (!result.ok) return { ok: false, id, file: path, problem: result.problem }
        if (result.run.id !== id) return { ok: false, id, file: path, problem: 'Run id does not match its file name.' }
        return { ok: true, run: result.run.status === 'running' ? { ...result.run, status: 'stopped' } : result.run }
      } catch (err) { return { ok: false, id, file: path, problem: err instanceof Error ? err.message : 'Could not read run.' } }
    }))
  }
  async save(raw: WritingBenchRun): Promise<WritingBenchRun> {
    const result = cleanRun(raw)
    if (!result.ok) throw new Error(result.problem)
    await writeJsonAtomic(this.path(result.run.id), result.run)
    return result.run
  }
  async delete(id: string): Promise<void> { await fs.rm(this.path(id), { force: true }) }
  async prompts(): Promise<BenchPrompt[]> {
    try { return cleanPrompts(JSON.parse((await fs.readFile(this.promptFile, 'utf8')).replace(/^\uFEFF/, ''))) } catch { return [] }
  }
  async savePrompts(prompts: BenchPrompt[]): Promise<BenchPrompt[]> {
    const clean = cleanPrompts(prompts)
    await readJson<unknown>(this.promptFile, [])
    await writeJsonAtomic(this.promptFile, clean)
    return clean
  }
}
