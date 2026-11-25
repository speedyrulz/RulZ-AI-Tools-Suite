// Runs PDF jobs in worker threads so a huge or hostile file can hang or crash only its own thread, never the app.
// Kept free of the worker's path (the main process passes it in) so tests can load this file.
import { Worker, type ResourceLimits } from 'node:worker_threads'
import { join } from 'node:path'
import type { PdfJob, PdfJobResult } from './extract'
import { PdfProblem } from './text'
import type { PdfWorkerOutput } from './worker'

/** Something that can read a PDF. The app uses the worker reader; tests use one that reads in their own thread. */
export interface PdfReader {
  /** Reads `data` (which the reader may take over) as the job says. Rejects with a PdfProblem. */
  run(data: Uint8Array, job: PdfJob, signal?: AbortSignal): Promise<PdfJobResult>
}

export const PDF_TIMEOUT_MS = 60_000
/** Resolve the same character maps in development and in the packaged app. */
export function pdfCmapDir(packaged: boolean, resourcesPath: string, mainDir: string): string {
  return packaged ? join(resourcesPath, 'pdfjs-cmaps') : join(mainDir, '../../node_modules/pdfjs-dist/cmaps')
}
const MAX_AT_ONCE = 2
/** The thread limits cover the JavaScript heap only, so growth of the whole app's memory during a read is watched too. */
const MAX_GROWTH_BYTES = 3 * 1024 ** 3
const MEMORY_CHECK_MS = 250
const MEMORY_MESSAGE = 'This PDF needs more memory than a read may use, so it was stopped. Try a few pages with pages.'

export interface WorkerReaderOptions {
  /** The built worker file (out/main/pdfWorker.js). */
  workerPath: string
  /** pdf.js character maps folder; looked up at each read. */
  cmapDir?: () => string | undefined
  timeoutMs?: number
  maxAtOnce?: number
  /** Memory limits of each thread (tests lower them). */
  resourceLimits?: ResourceLimits
  /** How far the app's memory may grow during a read (tests lower it). */
  maxGrowthBytes?: number
}

/** One worker thread per read, at most two at once; the rest wait their turn and can be stopped while waiting. */
export class PdfWorkerReader implements PdfReader {
  private active = 0
  private readonly waiting: (() => void)[] = []

  constructor(private readonly opts: WorkerReaderOptions) {}

  async run(data: Uint8Array, job: PdfJob, signal?: AbortSignal): Promise<PdfJobResult> {
    await this.slot(signal)
    let worker: { result: Promise<PdfJobResult>; ended: Promise<unknown> }
    try {
      worker = this.runInWorker(data, job, signal)
    } catch (e) {
      this.release()
      throw e
    }
    // The answer does not wait for the thread to end, but its place does, so there are never more threads than places.
    void worker.ended.then(() => this.release())
    return worker.result
  }

  private release(): void {
    this.active--
    this.waiting.shift()?.()
  }

  /** Waits for a free place. A stop while waiting leaves the line without taking one. */
  private slot(signal?: AbortSignal): Promise<void> {
    if (signal?.aborted) return Promise.reject(new PdfProblem('stopped', 'The read was stopped.'))
    if (this.active < (this.opts.maxAtOnce ?? MAX_AT_ONCE)) {
      this.active++
      return Promise.resolve()
    }
    return new Promise((resolve, reject) => {
      const turn = (): void => {
        signal?.removeEventListener('abort', onAbort)
        this.active++
        resolve()
      }
      const onAbort = (): void => {
        const i = this.waiting.indexOf(turn)
        if (i >= 0) this.waiting.splice(i, 1)
        reject(new PdfProblem('stopped', 'The read was stopped.'))
      }
      this.waiting.push(turn)
      signal?.addEventListener('abort', onAbort, { once: true })
    })
  }

  private runInWorker(data: Uint8Array, job: PdfJob, signal?: AbortSignal): { result: Promise<PdfJobResult>; ended: Promise<unknown> } {
    // A view into a shared buffer (a small file read from Node's pool) must not be handed over: copy it out.
    const own = data.byteOffset === 0 && data.byteLength === data.buffer.byteLength ? data : Uint8Array.from(data)
    const worker = new Worker(this.opts.workerPath, {
      workerData: { data: own, job, cmapDir: this.opts.cmapDir?.() },
      transferList: [own.buffer as ArrayBuffer],
      // A file that makes pdf.js eat memory ends this thread with an error instead of taking the app down.
      resourceLimits: this.opts.resourceLimits ?? { maxOldGenerationSizeMb: 1024, maxYoungGenerationSizeMb: 64, stackSizeMb: 4 },
      stdout: true,
      stderr: true
    })
    // Every way a read ends also ends the thread, and 'exit' always follows.
    const ended = new Promise((resolve) => worker.once('exit', resolve))
    const result = new Promise<PdfJobResult>((resolve, reject) => {
      let done = false
      const before = process.memoryUsage.rss()
      const finish = (fn: () => void): void => {
        if (done) return
        done = true
        clearTimeout(timer)
        clearInterval(memory)
        signal?.removeEventListener('abort', onAbort)
        void worker.terminate()
        fn()
      }
      const onAbort = (): void => finish(() => reject(new PdfProblem('stopped', 'The read was stopped.')))
      const timer = setTimeout(
        () => finish(() => reject(new PdfProblem('timeout', `Reading this PDF took longer than ${Math.round((this.opts.timeoutMs ?? PDF_TIMEOUT_MS) / 1000)} seconds, so it was stopped. Try a few pages with pages.`))),
        this.opts.timeoutMs ?? PDF_TIMEOUT_MS
      )
      timer.unref()
      const memory = setInterval(() => {
        if (process.memoryUsage.rss() - before > (this.opts.maxGrowthBytes ?? MAX_GROWTH_BYTES)) finish(() => reject(new PdfProblem('crash', MEMORY_MESSAGE)))
      }, MEMORY_CHECK_MS)
      memory.unref()
      signal?.addEventListener('abort', onAbort, { once: true })
      worker.on('message', (m: PdfWorkerOutput) =>
        finish(() => (m.ok ? resolve(m.result) : reject(new PdfProblem(m.code, m.message))))
      )
      worker.on('error', (e: Error & { code?: string }) =>
        finish(() =>
          reject(
            new PdfProblem(
              'crash',
              e.code === 'ERR_WORKER_OUT_OF_MEMORY'
                ? MEMORY_MESSAGE
                : `Reading this PDF failed: ${e.message}`
            )
          )
        )
      )
      worker.on('exit', (code) => finish(() => reject(new PdfProblem('crash', `The PDF reader stopped unexpectedly (code ${code}).`))))
      if (signal?.aborted) onAbort()
    })
    return { result, ended }
  }
}
