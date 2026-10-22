import { spawn, type ChildProcess } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { dirname } from 'node:path'
import type { InstanceErrorEvent, LoadConfig, LoadedInstance, MemoryEstimate, ModelEntry } from '@shared/types'
import { readJson, writeJsonAtomic } from '../util/jsonFile'
import { buildServerArgs } from './args'
import type { EngineService } from './engine'
import { engineEnv, freePort, isLlamaServerPid, killTree } from './process'
import { LogLineSplitter, nextProgress, progressFromDots } from './progress'

const MAX_LOG_LINES = 3000
const LOAD_TIMEOUT_MS = 20 * 60 * 1000

interface Running {
  info: LoadedInstance
  proc: ChildProcess
  log: string[]
  ready: Promise<LoadedInstance>
  exited: boolean
  exitCode: number | null
}

export interface InstanceManagerDeps {
  engine: EngineService
  getModel: (id: string) => ModelEntry | undefined
  getDefaults: (id: string) => LoadConfig
  unloadPreviousOnLoad: () => boolean
  estimate: (model: ModelEntry, config: LoadConfig) => MemoryEstimate | null
  pidFile: string
  onChange: (instances: LoadedInstance[]) => void
  onError: (event: InstanceErrorEvent) => void
}

function sameConfig(a: LoadConfig, b: LoadConfig): boolean {
  return JSON.stringify(a) === JSON.stringify(b)
}

/** Runs one llama-server process per loaded model, each on its own localhost port. */
export class InstanceManager {
  private running = new Map<string, Running>()

  constructor(private readonly deps: InstanceManagerDeps) {}

  list(): LoadedInstance[] {
    return [...this.running.values()].map((r) => ({ ...r.info }))
  }

  getLog(instanceId: string): string[] {
    return this.running.get(instanceId)?.log.slice() ?? []
  }

  /** The ready instance serving `modelId`, if any. */
  readyFor(modelId: string): LoadedInstance | undefined {
    for (const r of this.running.values()) if (r.info.modelId === modelId && r.info.status === 'ready') return r.info
    return undefined
  }

  baseUrl(instance: LoadedInstance): string {
    return `http://127.0.0.1:${instance.port}`
  }

  async load(modelId: string, config: LoadConfig | null): Promise<LoadedInstance> {
    const model = this.deps.getModel(modelId)
    if (!model) throw new Error(`Model not found: ${modelId}`)
    const cfg = config ?? this.deps.getDefaults(modelId)

    const existing = [...this.running.values()].find((r) => r.info.modelId === modelId && r.info.status !== 'stopping')
    if (existing && sameConfig(existing.info.config, cfg)) return existing.ready
    if (existing) await this.unload(existing.info.instanceId)
    if (this.deps.unloadPreviousOnLoad()) await this.unloadAll()

    const exe = await this.deps.engine.resolveExe()
    const port = await freePort()
    const instanceId = randomUUID()
    const args = buildServerArgs({ model, config: cfg, port, alias: model.name })
    const proc = spawn(exe, args, {
      cwd: dirname(exe),
      env: engineEnv(),
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe']
    })

    const info: LoadedInstance = {
      instanceId,
      modelId,
      modelName: model.name,
      port,
      pid: proc.pid ?? null,
      status: 'loading',
      progress: 0.02,
      stage: 'Starting',
      config: cfg,
      startedAt: Date.now(),
      contextLength: null,
      hasVision: false,
      estimate: this.deps.estimate(model, cfg)
    }
    const run: Running = { info, proc, log: [`$ llama-server ${args.join(' ')}`], ready: null!, exited: false, exitCode: null }
    this.running.set(instanceId, run)
    this.attachOutput(run)
    run.ready = this.waitReady(run)
    // Avoid an unhandled rejection when nobody awaits a failed background load.
    run.ready.catch(() => undefined)
    void this.savePids()
    this.emit()
    return run.ready
  }

  private attachOutput(run: Running): void {
    const splitter = new LogLineSplitter()
    let lastEmit = 0
    const onData = (chunk: Buffer): void => {
      const { lines, dots } = splitter.push(chunk.toString())
      let p = { progress: run.info.progress, stage: run.info.stage }
      for (const line of lines) {
        run.log.push(line)
        if (run.info.status === 'loading') p = nextProgress(p, line)
      }
      if (run.log.length > MAX_LOG_LINES) run.log.splice(0, run.log.length - MAX_LOG_LINES)
      if (run.info.status === 'loading' && dots !== null) {
        const fromDots = progressFromDots(dots)
        if (fromDots > p.progress) p = { progress: fromDots, stage: 'Loading weights' }
      }
      if (run.info.status === 'loading' && p.progress !== run.info.progress) {
        run.info.progress = p.progress
        run.info.stage = p.stage
        const now = Date.now()
        if (now - lastEmit > 150) {
          lastEmit = now
          this.emit()
        }
      }
    }
    run.proc.stdout?.on('data', onData)
    run.proc.stderr?.on('data', onData)

    run.proc.on('error', (err) => {
      run.log.push(`[failed to start] ${err.message}`)
      this.onExit(run, null)
    })
    run.proc.on('exit', (code) => {
      run.log.push(...splitter.flush(), `[process exited with code ${code}]`)
      this.onExit(run, code)
    })
  }

  private onExit(run: Running, code: number | null): void {
    if (run.exited) return
    run.exited = true
    run.exitCode = code
    const wasReady = run.info.status === 'ready'
    const expected = run.info.status === 'stopping'
    this.running.delete(run.info.instanceId)
    void this.savePids()
    this.emit()
    if (wasReady && !expected) {
      this.deps.onError({
        modelName: run.info.modelName,
        error: `llama-server stopped unexpectedly (exit code ${code}).`,
        logTail: run.log.slice(-40)
      })
    }
  }

  private async waitReady(run: Running): Promise<LoadedInstance> {
    const base = this.baseUrl(run.info)
    const deadline = Date.now() + LOAD_TIMEOUT_MS
    while (Date.now() < deadline) {
      if (run.exited) {
        if (run.info.status === 'stopping') throw new Error('Load cancelled')
        throw new Error(this.failureMessage(run))
      }
      if (run.info.status === 'stopping') throw new Error('Load cancelled')
      try {
        const res = await fetch(`${base}/health`, { signal: AbortSignal.timeout(2000) })
        if (res.ok) break
      } catch {
        // not listening yet
      }
      await new Promise((r) => setTimeout(r, 400))
    }
    if (Date.now() >= deadline) {
      await this.unload(run.info.instanceId)
      throw new Error('Timed out waiting for the model to load.')
    }

    try {
      const props = (await (await fetch(`${base}/props`, { signal: AbortSignal.timeout(5000) })).json()) as {
        n_ctx?: number
        default_generation_settings?: { n_ctx?: number }
        modalities?: { vision?: boolean }
      }
      run.info.contextLength = props.default_generation_settings?.n_ctx ?? props.n_ctx ?? null
      run.info.hasVision = !!props.modalities?.vision
    } catch {
      // props are informational only
    }
    run.info.status = 'ready'
    run.info.progress = 1
    run.info.stage = 'Ready'
    this.emit()
    return { ...run.info }
  }

  private failureMessage(run: Running): string {
    const errLines = run.log.filter((l) => /error|failed|out of memory|unable|cannot|invalid/i.test(l)).slice(-6)
    const detail = errLines.length ? errLines.join('\n') : run.log.slice(-6).join('\n')
    return `llama-server exited while loading ${run.info.modelName} (code ${run.exitCode}).\n${detail}`
  }

  async unload(instanceId: string): Promise<void> {
    const run = this.running.get(instanceId)
    if (!run) return
    run.info.status = 'stopping'
    run.info.stage = 'Unloading'
    this.emit()
    const exited = new Promise<void>((resolve) => {
      if (run.exited) return resolve()
      run.proc.once('exit', () => resolve())
      setTimeout(resolve, 8000)
    })
    killTree(run.proc.pid)
    await exited
    if (!run.exited) this.onExit(run, null)
  }

  async unloadAll(): Promise<void> {
    await Promise.all([...this.running.keys()].map((id) => this.unload(id)))
  }

  /** Synchronous kill for app shutdown. */
  killAllSync(): void {
    for (const run of this.running.values()) {
      run.info.status = 'stopping'
      killTree(run.proc.pid, true)
    }
    this.running.clear()
    void this.savePids()
  }

  /** Kills llama-server processes left behind if the app crashed last time. */
  async cleanupOrphans(): Promise<void> {
    const pids = await readJson<number[]>(this.deps.pidFile, [])
    for (const pid of pids) if (isLlamaServerPid(pid)) killTree(pid, true)
    await this.savePids()
  }

  private savePids(): Promise<void> {
    const pids = [...this.running.values()].map((r) => r.proc.pid).filter((p): p is number => !!p)
    return writeJsonAtomic(this.deps.pidFile, pids).catch(() => undefined)
  }

  private emit(): void {
    this.deps.onChange(this.list())
  }
}
