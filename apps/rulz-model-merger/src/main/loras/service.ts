// LoRA + LoRA merges computed by the app itself (worker thread, CPU). Only one merge runs at a time.
//
// Files: the worker writes into a private staging folder `<baseDir>/lora-merge/<jobId>/` inside WSL (through the UNC
// path). When it finishes, the result and its `.recipe.json` sidecar are moved with `mv -n` into `<models>/loras/merged/`
// under a free name. The service only ever deletes files inside its own `<jobId>` staging folder (two files and an
// rmdir of the then-empty folder); it never deletes recursively and never touches the models folder except for the move.
import { EventEmitter } from 'node:events'
import { mkdir, rmdir, unlink, writeFile } from 'node:fs/promises'
import { Worker } from 'node:worker_threads'
import { MAX_LORA_MERGE_INPUTS } from '@shared/loraMerge'
import type {
  LoraMergeCheck,
  LoraMergeInspectRequest,
  LoraMergeJob,
  LoraMergeRecipe,
  LoraMergeRecord
} from '@shared/loraMerge'
import type { OwnServerSettings } from '@shared/types'
import { sanitizeOutputName } from '../merge/graph'
import { moveNoOverwrite, resolveModelsDir, sourcePath, toUncPath } from '../merge/wslFiles'
import { readSafetensorsHeader, type SafetensorsHeader } from '../models/safetensors'
import type { WslRunner } from '../server/wsl'
import { listLoras, resolveStagingDir } from './files'
import { checkLoraMerge, describeMergeInput } from './merge'
import type { LoraWorkerMessage, LoraWorkerRequest } from './workerTypes'

export interface WorkerHandle {
  /** Stops the worker; resolves once its thread has exited (so its files are closed). */
  terminate(): void | Promise<void>
  /** Resolves with the worker's last message ('done' or 'error'); also resolves (with an error) when it stops early. */
  done: Promise<LoraWorkerMessage>
}

export interface LoraMergeServiceDeps {
  wsl: WslRunner
  settings: () => OwnServerSettings
  appVersion: string
  /** Path of the bundled worker file (used by the default runWorker). */
  workerPath?: string
  /** Test hooks. */
  readHeader?: (uncPath: string) => Promise<SafetensorsHeader>
  runWorker?: (req: LoraWorkerRequest, onMessage: (m: LoraWorkerMessage) => void) => WorkerHandle
  newId?: () => string
  now?: () => number
  toUnc?: (distro: string, absWslPath: string) => string
  writeText?: (uncPath: string, text: string) => Promise<void>
  unlinkFile?: (uncPath: string) => Promise<void>
  removeDir?: (uncPath: string) => Promise<void>
  makeDir?: (uncPath: string) => Promise<void>
  /** Minimum milliseconds between progress events. Default 200. */
  progressMs?: number
}

interface Active {
  job: LoraMergeJob
  handle: WorkerHandle | null
  cancelled: boolean
  /** Settles when the run has cleaned up. */
  finished: Promise<void>
  jobDirUnc: string
  files: string[]
}

const ENOENT_OK = async (fn: () => Promise<void>): Promise<void> => {
  try {
    await fn()
  } catch {
    // Already gone, or still held by a stopping worker: nothing else to do for a partial file.
  }
}

export function defaultRunWorker(workerPath: string): NonNullable<LoraMergeServiceDeps['runWorker']> {
  return (req, onMessage) => {
    const worker = new Worker(workerPath, { workerData: req })
    let terminating = false
    const done = new Promise<LoraWorkerMessage>((resolve) => {
      let last: LoraWorkerMessage | null = null
      worker.on('message', (m: LoraWorkerMessage) => {
        if (m.type === 'progress') onMessage(m)
        else {
          last = m
          resolve(m)
        }
      })
      worker.on('error', (e) => resolve({ type: 'error', message: e instanceof Error ? e.message : String(e) }))
      worker.on('exit', (code) => {
        if (last) return
        resolve({ type: 'error', message: terminating ? 'The merge was stopped.' : `The merge worker stopped unexpectedly (code ${code}).` })
      })
    })
    return {
      terminate: async () => {
        terminating = true
        await worker.terminate().catch(() => undefined)
      },
      done
    }
  }
}

export class LoraMergeService extends EventEmitter {
  private readonly list_: LoraMergeJob[] = []
  private active: Active | null = null
  private busy = false

  constructor(private readonly d: LoraMergeServiceDeps) {
    super()
  }

  private now(): number {
    return (this.d.now ?? Date.now)()
  }

  private unc(distro: string, p: string): string {
    return (this.d.toUnc ?? toUncPath)(distro, p)
  }

  async list(): Promise<string[]> {
    const s = this.d.settings()
    const modelsAbs = await resolveModelsDir(this.d.wsl, s.distro, s)
    return listLoras(this.d.wsl, s.distro, modelsAbs)
  }

  jobs(): LoraMergeJob[] {
    return this.list_.map((j) => ({ ...j })).sort((a, b) => b.createdAt - a.createdAt)
  }

  private async readInputs(
    modelsAbs: string,
    distro: string,
    names: string[],
    check: {
      mode: LoraMergeRecipe['mode']
      rank: number
      dtype: LoraMergeRecipe['dtype']
      weights?: number[]
    }
  ): Promise<{ check: LoraMergeCheck; headers: Array<SafetensorsHeader | null> }> {
    const read = this.d.readHeader ?? readSafetensorsHeader
    const headers: Array<SafetensorsHeader | null> = []
    const infos = []
    for (const name of names) {
      try {
        const header = await read(this.unc(distro, sourcePath(modelsAbs, 'lora', name)))
        headers.push(header)
        infos.push(describeMergeInput(name, header, null))
      } catch (e) {
        headers.push(null)
        infos.push(describeMergeInput(name, null, (e as Error).message))
      }
    }
    return { check: checkLoraMerge(infos, headers, check), headers }
  }

  async inspect(req: LoraMergeInspectRequest): Promise<LoraMergeCheck> {
    const s = this.d.settings()
    const modelsAbs = await resolveModelsDir(this.d.wsl, s.distro, s)
    const r = await this.readInputs(modelsAbs, s.distro, req.names, { mode: req.mode, rank: req.rank, dtype: req.dtype })
    return r.check
  }

  private validate(recipe: LoraMergeRecipe): string | null {
    const n = recipe.inputs?.length ?? 0
    if (n < 2) return 'Choose at least two LoRAs to merge.'
    if (n > MAX_LORA_MERGE_INPUTS) return `At most ${MAX_LORA_MERGE_INPUTS} LoRAs can be merged at once.`
    for (const i of recipe.inputs) {
      if (typeof i.name !== 'string' || !i.name.trim()) return 'Every LoRA needs a file name.'
      if (typeof i.weight !== 'number' || !Number.isFinite(i.weight) || i.weight < -10 || i.weight > 10) {
        return `The weight of "${i.name}" must be a number from -10 to 10.`
      }
    }
    if (recipe.mode !== 'concat' && recipe.mode !== 'shrink') return 'The merge mode must be concat or shrink.'
    if (!['source', 'bf16', 'fp16', 'fp32'].includes(recipe.dtype)) return 'The data type must be source, bf16, fp16 or fp32.'
    if (!Number.isInteger(recipe.rank) || recipe.rank < 1 || recipe.rank > 1024) return 'The rank must be a whole number from 1 to 1024.'
    if (typeof recipe.outputName !== 'string' || !recipe.outputName.trim()) return 'Give the merged LoRA a name.'
    return null
  }

  async start(recipeIn: LoraMergeRecipe): Promise<LoraMergeJob> {
    if (this.busy) throw new Error('A LoRA merge is already running.')
    this.busy = true
    try {
      const problem = this.validate(recipeIn)
      if (problem) throw new Error(problem)
      const stemRaw = recipeIn.outputName.trim()
      const stem = sanitizeOutputName(stemRaw)
      // sanitizeOutputName falls back to "merged" for an empty result; a name with no letter or digit is refused instead.
      if (!/[A-Za-z0-9]/.test(stemRaw)) throw new Error('The name has no usable letters or digits.')
      const recipe: LoraMergeRecipe = {
        inputs: recipeIn.inputs.map((i) => ({ name: i.name.trim(), weight: i.weight })),
        mode: recipeIn.mode,
        rank: recipeIn.rank,
        dtype: recipeIn.dtype,
        outputName: stem
      }
      const s = this.d.settings()
      const distro = s.distro
      const modelsAbs = await resolveModelsDir(this.d.wsl, distro, s)
      const { check, headers } = await this.readInputs(
        modelsAbs,
        distro,
        recipe.inputs.map((i) => i.name),
        { mode: recipe.mode, rank: recipe.rank, dtype: recipe.dtype, weights: recipe.inputs.map((i) => i.weight) }
      )
      if (check.problems.length) throw new Error(check.problems.join(' '))
      const staging = await resolveStagingDir(this.d.wsl, distro, s.baseDir)
      const id = (this.d.newId ?? (() => `lora-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`))()
      const jobDir = `${staging}/${id}`
      const outPath = `${jobDir}/${stem}.safetensors`
      const job: LoraMergeJob = {
        id,
        recipe,
        status: 'running',
        createdAt: this.now(),
        finishedAt: null,
        progress: null,
        error: null,
        output: null
      }
      const active: Active = {
        job,
        handle: null,
        cancelled: false,
        finished: Promise.resolve(),
        jobDirUnc: this.unc(distro, jobDir),
        files: [this.unc(distro, outPath), this.unc(distro, `${outPath}.recipe.json`)]
      }
      this.active = active
      this.list_.push(job)
      this.emitJob(job)
      active.finished = this.run(active, {
        distro,
        modelsAbs,
        jobDir,
        outPath,
        headerShas: headers.map((h) => h?.headerSha256 ?? null)
      }).finally(() => {
        if (this.active === active) this.active = null
        this.busy = false
      })
      return { ...job }
    } catch (e) {
      // Not started: the busy flag is only kept while a job runs (released by run()).
      if (!this.active) this.busy = false
      throw e
    }
  }

  private emitJob(job: LoraMergeJob): void {
    this.emit('job', { ...job })
  }

  private async cleanup(a: Active): Promise<void> {
    const unlinkFile = this.d.unlinkFile ?? unlink
    const removeDir = this.d.removeDir ?? rmdir
    for (const f of a.files) await ENOENT_OK(() => unlinkFile(f))
    await ENOENT_OK(() => removeDir(a.jobDirUnc))
  }

  private async fail(a: Active, message: string): Promise<void> {
    await this.cleanup(a)
    a.job.status = 'error'
    a.job.error = message
    a.job.finishedAt = this.now()
    this.emitJob(a.job)
  }

  private async run(
    a: Active,
    ctx: { distro: string; modelsAbs: string; jobDir: string; outPath: string; headerShas: Array<string | null> }
  ): Promise<void> {
    const { job } = a
    const makeDir = this.d.makeDir ?? (async (p: string) => void (await mkdir(p, { recursive: true })))
    try {
      // The staging folder exists already (resolveStagingDir made `lora-merge`); the job folder is made here.
      await makeDir(a.jobDirUnc)
      if (a.cancelled) {
        await this.cleanup(a)
        return
      }
      const req: LoraWorkerRequest = {
        inputs: job.recipe.inputs.map((i) => ({
          name: i.name,
          uncPath: this.unc(ctx.distro, sourcePath(ctx.modelsAbs, 'lora', i.name)),
          weight: i.weight
        })),
        recipe: job.recipe,
        outUncPath: a.files[0],
        appVersion: this.d.appVersion,
        inputHeaderSha256: ctx.headerShas
      }
      const runWorker = this.d.runWorker ?? defaultRunWorker(this.d.workerPath ?? '')
      let lastEmit = 0
      const throttle = this.d.progressMs ?? 200
      a.handle = runWorker(req, (m) => {
        if (m.type !== 'progress' || a.cancelled) return
        job.progress = { done: m.done, total: m.total }
        const t = this.now()
        if (m.done >= m.total || t - lastEmit >= throttle) {
          lastEmit = t
          this.emitJob(job)
        }
      })
      const result = await a.handle.done
      if (a.cancelled) return
      if (result.type === 'progress') throw new Error('The merge worker ended without a result.')
      if (result.type === 'error') throw new Error(result.message)

      job.status = 'saving'
      job.progress = null
      this.emitJob(job)
      const record = this.recordFrom(result.metadata, job.recipe, ctx.headerShas)
      const sidecarUnc = a.files[1]
      await (this.d.writeText ?? ((p, t) => writeFile(p, t, 'utf8')))(sidecarUnc, JSON.stringify(record, null, 2) + '\n')
      const lorasDir = `${ctx.modelsAbs.replace(/\/+$/, '')}/loras`
      const moved = await moveNoOverwrite(this.d.wsl, ctx.distro, ctx.outPath, `${lorasDir}/merged`)
      const removeDir = this.d.removeDir ?? rmdir
      // Empty now (both files moved); rmdir only, so a folder that still holds anything is left alone.
      await ENOENT_OK(() => removeDir(a.jobDirUnc))
      const loraName = moved.path.startsWith(`${lorasDir}/`) ? moved.path.slice(lorasDir.length + 1) : `merged/${moved.path.split('/').pop()}`
      // A verify with ok=false still moves the file: the user may want it anyway. The job is "done" and the UI warns.
      job.output = {
        wslPath: moved.path,
        loraName,
        sizeBytes: result.sizeBytes,
        modules: result.modules,
        rank: result.rank,
        energyKept: result.energyKept,
        verify: result.verify,
        sidecarPath: moved.sidecar
      }
      job.status = 'done'
      job.finishedAt = this.now()
      this.emitJob(job)
    } catch (e) {
      if (a.cancelled) return
      await this.fail(a, (e as Error).message || 'The merge failed.')
    }
  }

  /** The record saved in the file's metadata when it parses, else built from the recipe. */
  private recordFrom(metadata: Record<string, string>, recipe: LoraMergeRecipe, shas: Array<string | null>): LoraMergeRecord {
    try {
      const parsed = JSON.parse(metadata.rulz_model_merger ?? '') as LoraMergeRecord
      if (parsed && parsed.kind === 'lora-merge') return parsed
    } catch {
      // fall through
    }
    return {
      app: 'RulZ Model Merger',
      appVersion: this.d.appVersion,
      kind: 'lora-merge',
      createdAt: new Date(this.now()).toISOString(),
      recipe,
      inputHeaderSha256: shas
    }
  }

  async cancel(id: string): Promise<LoraMergeJob> {
    const a = this.active
    if (!a || a.job.id !== id) {
      const j = this.list_.find((x) => x.id === id)
      if (!j) throw new Error('That merge is not known.')
      return { ...j }
    }
    if (a.job.status === 'saving') throw new Error('The merge is being saved and can no longer be cancelled.')
    a.cancelled = true
    await a.handle?.terminate()
    // Let a stopping worker release its files, then remove only this job's partial output.
    await a.handle?.done.catch(() => undefined)
    await this.cleanup(a)
    a.job.status = 'cancelled'
    a.job.progress = null
    a.job.finishedAt = this.now()
    this.emitJob(a.job)
    await a.finished
    return { ...a.job }
  }
}
