// Saved practice batch runs (the scoreboard), one JSON file in the app's data folder.
import { cleanBatchRun, settleRun, type BatchRun } from '@shared/batch'
import { readJson, writeJsonAtomic } from '../util/jsonFile'

/** Older runs beyond this are dropped when a new one is saved. */
export const MAX_BATCH_RUNS = 200

export class BatchStore {
  private runs: BatchRun[] | null = null
  /** Reads, saves and deletes run one after another, so two saves cannot write each other's list away. */
  private queue: Promise<unknown> = Promise.resolve()

  constructor(
    private readonly path: string,
    private readonly maxRuns = MAX_BATCH_RUNS
  ) {}

  private enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const next = this.queue.catch(() => undefined).then(fn)
    this.queue = next
    return next
  }

  private async load(): Promise<BatchRun[]> {
    if (!this.runs) {
      const raw = await readJson<unknown>(this.path, [])
      const list = Array.isArray(raw) ? raw : []
      // Nothing is running when the file is first read: a run still marked so was cut off.
      this.runs = list.flatMap((r) => {
        const run = cleanBatchRun(r)
        return run ? [settleRun(run)] : []
      })
    }
    return this.runs
  }

  /**
   * Newest first. Only a window that is starting asks for the list, and the run loop lives in the
   * window: so a run still marked running here was cut off (the window was reloaded or crashed).
   */
  list(): Promise<BatchRun[]> {
    return this.enqueue(async () => {
      this.runs = (await this.load()).map(settleRun)
      return [...this.runs].sort((a, b) => b.createdAt - a.createdAt)
    })
  }

  save(raw: unknown): Promise<void> {
    return this.enqueue(async () => {
      const run = cleanBatchRun(raw)
      if (!run) throw new Error('Not a batch run.')
      const runs = (await this.load()).filter((r) => r.id !== run.id)
      runs.push(run)
      runs.sort((a, b) => b.createdAt - a.createdAt)
      this.runs = runs.slice(0, this.maxRuns)
      await writeJsonAtomic(this.path, this.runs)
    })
  }

  delete(id: string): Promise<void> {
    return this.enqueue(async () => {
      const runs = await this.load()
      if (!runs.some((r) => r.id === id)) return
      this.runs = runs.filter((r) => r.id !== id)
      await writeJsonAtomic(this.path, this.runs)
    })
  }
}
