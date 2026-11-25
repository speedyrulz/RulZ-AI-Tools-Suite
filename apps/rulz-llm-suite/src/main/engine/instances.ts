import type { ChildProcess } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { adapterPathUsable } from '@shared/adapters'
import type { AppMode } from '@shared/modes'
import type { RemoteHost } from '@shared/remote'
import type { AdapterEntry, InstanceAdapter, InstanceErrorEvent, LoadAdapter, LoadConfig, LoadedInstance, MemoryEstimate, ModelEntry, SidecarFile } from '@shared/types'
import { buildServerArgs } from './args'
import type { EngineService } from './engine'
import { killOrphanProcesses, savePidFile, startLocalServer } from './localServer'
import { describeExitCode, describeLoadLog, freePort } from './process'
import { LogLineSplitter, nextProgress, progressFromDots } from './progress'
import type { RemoteLaunch, RemoteLauncher, ServerHandle } from './serverHandle'

const MAX_LOG_LINES = 3000
const LOAD_TIMEOUT_MS = 20 * 60 * 1000
/** A big model can take long to read from a rental's disk, and a dropped tunnel is waited out. */
const REMOTE_LOAD_TIMEOUT_MS = 60 * 60 * 1000
/** A health check through the tunnel can take a while; a short timeout would read a slow answer as "not up". */
const REMOTE_HEALTH_TIMEOUT_MS = 10_000
const RECONNECT_POLL_MS = 2000

type UseBy = 'window' | 'api' | 'phone'

interface Running {
  info: LoadedInstance
  /** Null only while a remote server is still being started (nothing to stop yet). */
  handle: ServerHandle | null
  host?: RemoteHost
  log: string[]
  ready: Promise<LoadedInstance>
  exited: boolean
  exitCode: number | null
  /** The sentence the handle gave for its exit, if any. */
  exitReason?: string
  /** Requests or chats using the instance right now; idle-unload leaves it alone while above 0. */
  users: number
  /** When it became ready or was last released by a user. */
  lastUsedAt: number
  /** Unloaded (or the app quit) before its remote server was up; a handle that arrives later is stopped. */
  cancelled: boolean
  cancel: { promise: Promise<never>; reject: (err: Error) => void }
  /** An unload of a remote instance in progress, so a second call joins it. */
  unloading?: Promise<void>
}

/** Where an instance runs, as a map key: loads, queues and "unload the previous model" are per host. */
const hostKey = (host?: RemoteHost): string => (host ? `${host.kind}:${host.rentalId}` : 'local')

export interface InstanceManagerDeps {
  engine: EngineService
  getModel: (id: string) => ModelEntry | undefined
  /** The draft this load uses (the chosen one, or the model's own); undefined = none. */
  draftFor: (model: ModelEntry, cfg: LoadConfig) => SidecarFile | undefined
  /** The load settings of a model, for a mode when given (that mode's own, else the model's defaults). */
  getDefaults: (id: string, mode?: AppMode) => LoadConfig
  unloadPreviousOnLoad: () => boolean
  estimate: (model: ModelEntry, config: LoadConfig) => MemoryEstimate | null
  /** An adapter file by id, for the adapters a load config names. */
  getAdapter?: (id: string) => AdapterEntry | undefined
  pidFile: string
  onChange: (instances: LoadedInstance[]) => void
  onError: (event: InstanceErrorEvent) => void
  /** Called once an instance's process is gone, so per-instance caches elsewhere can be dropped. */
  onGone?: (instanceId: string) => void
  /** Starts llama-server; injectable for tests. */
  spawnServer?: (exe: string, args: string[]) => ChildProcess
  /** Kills a server's process tree (`force` = synchronous); injectable for tests. */
  killProcess?: (proc: ChildProcess, force: boolean) => void
  /** Overrides the unload waits [first, after forcing] in ms; for tests. */
  unloadWaitMs?: [number, number]
  /** Starts llama-server on a vast.ai rental; without it remote models cannot be loaded. */
  launcher?: RemoteLauncher
  /**
   * Called for every acquire; the returned function ends that use. The rental side uses it to keep a rental
   * awake and to show what is using it (it should ignore instances without a `host`).
   */
  onUse?: (instance: LoadedInstance, by: UseBy, what: string) => () => void
  /** Milliseconds between checks of the tunnels of remote instances; for tests. */
  reconnectPollMs?: number
}

function sameConfig(a: LoadConfig, b: LoadConfig): boolean {
  // The adapter list is part of the config, so the same model with other adapters or strengths is a new load.
  return JSON.stringify(a) === JSON.stringify(b)
}

/**
 * The adapters a load passes to llama-server, in order, plus a log line for each one left out (not in
 * the library any more, or a path llama.cpp can't take). Their position is llama-server's adapter id.
 */
export function resolveAdapters(
  wanted: LoadAdapter[],
  getAdapter: (id: string) => AdapterEntry | undefined
): { loaded: InstanceAdapter[]; notes: string[] } {
  const loaded: InstanceAdapter[] = []
  const notes: string[] = []
  for (const w of wanted) {
    const a = getAdapter(w.id)
    if (!a) notes.push(`[adapter skipped] ${w.id}: not found in the models folder`)
    else if (!adapterPathUsable(a.path)) notes.push(`[adapter skipped] ${w.id}: llama.cpp can't load a path containing a comma; rename the file`)
    else loaded.push({ id: a.id, name: a.name, path: a.path, scale: w.scale })
  }
  return { loaded, notes }
}

/**
 * Runs one llama-server per loaded model: a process on this PC (its own localhost port) or, for a remote
 * model, a server on a vast.ai rental reached through a local tunnel port. Loads, queues and "unload the
 * previous model" are per host, so a rental never waits for this PC or the other way round.
 */
export class InstanceManager {
  private running = new Map<string, Running>()
  /**
   * Loads on one host take turns from "unload the previous model" to "process registered", so two overlapping
   * loads (UI + API) can't both pass unloadAll() and leave two models in VRAM. One queue per host.
   */
  private loadQueues = new Map<string, Promise<unknown>>()
  /** Loads queued or starting, so ensure() of a remote model finds the host a queued load is about to use. */
  private pending: { modelId: string; host?: RemoteHost }[] = []
  /** Set when the app is quitting; no server may be started after that. */
  private disposed = false
  private reconnectTimer: ReturnType<typeof setInterval> | null = null

  constructor(private readonly deps: InstanceManagerDeps) {}

  list(): LoadedInstance[] {
    return [...this.running.values()].map((r) => ({ ...r.info }))
  }

  getLog(instanceId: string): string[] {
    return this.running.get(instanceId)?.log.slice() ?? []
  }

  /** An instance by id, whatever its state; undefined once it is gone. */
  get(instanceId: string): LoadedInstance | undefined {
    return this.running.get(instanceId)?.info
  }

  /**
   * A ready instance serving `modelId`, if any. `prefer` wins when it is ready. Otherwise one whose connection is
   * not being re-made, then the one with the fewest users, then the newest.
   */
  readyFor(modelId: string, prefer?: string): LoadedInstance | undefined {
    const ready = [...this.running.values()].filter((r) => r.info.modelId === modelId && r.info.status === 'ready')
    if (!ready.length) return undefined
    if (prefer) {
      const hit = ready.find((r) => r.info.instanceId === prefer)
      if (hit) return hit.info
    }
    if (ready.length === 1) return ready[0].info
    const order = new Map(ready.map((r, i) => [r, i]))
    ready.sort(
      (a, b) =>
        Number(!!a.info.reconnecting) - Number(!!b.info.reconnecting) ||
        a.users - b.users ||
        b.info.startedAt - a.info.startedAt ||
        order.get(b)! - order.get(a)!
    )
    return ready[0].info
  }

  baseUrl(instance: LoadedInstance): string {
    return `http://127.0.0.1:${instance.port}`
  }

  /**
   * Loads a model. A remote model is loaded on `host`; without one, an instance of it that already exists is
   * reloaded on its own host, and when there is none the call fails (this never starts or rents anything).
   */
  load(modelId: string, config: LoadConfig | null, host?: RemoteHost): Promise<LoadedInstance> {
    return this.queueLoad(modelId, config, false, host).then((r) => r.instance)
  }

  /**
   * The model's running or loading instance whatever its settings, or else a new one with its saved
   * defaults; `started` says whether this call started it. It takes its turn in the load queue, so
   * a load the UI has already queued is found rather than replaced. A remote model with no instance is an
   * error: only a click on the vast.ai page starts one. A new instance gets the settings of `mode` when it has some.
   */
  ensure(modelId: string, mode?: AppMode): Promise<{ instance: LoadedInstance; started: boolean }> {
    return this.queueLoad(modelId, null, true, undefined, mode)
  }

  /** The host a load goes to: this PC (undefined) or a rental. Throws the sentence for a request that cannot be placed. */
  private resolveHost(modelId: string, host: RemoteHost | undefined): RemoteHost | undefined {
    const model = this.deps.getModel(modelId)
    // A missing model is reported by the load itself.
    if (!model) return host
    if (!model.remote) {
      if (host) throw new Error(`${model.name} runs on this PC, not on a rental.`)
      return undefined
    }
    if (host) return host
    const mine = [...this.running.values()].filter((r) => r.info.modelId === modelId && r.info.status !== 'stopping')
    mine.sort((a, b) => Number(b.info.status === 'ready') - Number(a.info.status === 'ready') || b.info.startedAt - a.info.startedAt)
    const found = mine[0]?.host ?? this.pending.find((p) => p.modelId === modelId)?.host
    if (!found) throw new Error(`Load ${model.name} on a rental first (vast.ai page).`)
    return found
  }

  private queueLoad(
    modelId: string,
    config: LoadConfig | null,
    reuseAny: boolean,
    host?: RemoteHost,
    mode?: AppMode
  ): Promise<{ instance: LoadedInstance; started: boolean }> {
    let target: RemoteHost | undefined
    try {
      target = this.resolveHost(modelId, host)
    } catch (err) {
      return Promise.reject(err)
    }
    const key = hostKey(target)
    const entry = { modelId, host: target }
    this.pending.push(entry)
    const started = (this.loadQueues.get(key) ?? Promise.resolve()).then(() => this.startLoad(modelId, config, reuseAny, target, mode))
    // The queue only waits for the start; a failed start must not block the next load.
    this.loadQueues.set(key, started.catch(() => undefined))
    const done = (): void => {
      this.pending = this.pending.filter((p) => p !== entry)
    }
    started.then(done, done)
    // Waiting for the model to become ready happens outside the queue, so a long load can still be
    // replaced (the next load unloads it) or cancelled.
    return started.then(async (s) => ({ instance: await s.ready, started: s.started }))
  }

  /**
   * Marks an instance as in use until the returned function is called, so idle-unload leaves it
   * alone however long the use takes (a long stream, a chat turn with tool calls). Calling the
   * returned function more than once is harmless. `by` and `what` tell a rental who is using it.
   */
  acquire(instanceId: string, by: UseBy = 'window', what = 'chat'): () => void {
    const run = this.running.get(instanceId)
    if (!run) return () => undefined
    run.users++
    let end: (() => void) | undefined
    try {
      end = this.deps.onUse?.({ ...run.info }, by, what)
    } catch {
      // Keeping a rental awake is a side job; it must never fail the request.
    }
    let released = false
    return () => {
      if (released) return
      released = true
      run.users--
      run.lastUsedAt = Date.now()
      try {
        end?.()
      } catch {
        // see above
      }
    }
  }

  /** When a ready instance was last used; null while it is in use, loading or gone. */
  idleSince(instanceId: string): number | null {
    const run = this.running.get(instanceId)
    if (!run || run.users > 0 || run.info.status !== 'ready') return null
    return run.lastUsedAt
  }

  private checkDisposed(): void {
    if (this.disposed) throw new Error('The app is closing.')
  }

  /**
   * Unloads what has to go and starts the server. The ready promise is wrapped so that awaiting the
   * start doesn't also wait for the model to load.
   */
  private async startLoad(
    modelId: string,
    config: LoadConfig | null,
    reuseAny: boolean,
    host: RemoteHost | undefined,
    mode?: AppMode
  ): Promise<{ ready: Promise<LoadedInstance>; started: boolean }> {
    this.checkDisposed()
    const found = this.deps.getModel(modelId)
    if (!found) throw new Error(`Model not found: ${modelId}`)
    // Refused before anything is unloaded: a request that cannot be placed must not cost a running model.
    if (found.remote && !host) throw new Error(`Load ${found.name} on a rental first (vast.ai page).`)
    if (!found.remote && host) throw new Error(`${found.name} runs on this PC, not on a rental.`)
    const cfg = config ?? this.deps.getDefaults(modelId, mode)
    const model = { ...found, draft: this.deps.draftFor(found, cfg) }
    const key = hostKey(host)

    const existing = [...this.running.values()].find((r) => r.info.modelId === modelId && hostKey(r.host) === key && r.info.status !== 'stopping')
    if (existing && (reuseAny || sameConfig(existing.info.config, cfg))) return { ready: existing.ready, started: false }
    // Quitting during any of these waits must not leave a llama-server running after the app is gone.
    if (existing) await this.unload(existing.info.instanceId)
    this.checkDisposed()
    // Only the previous models of the same host go: a rental's model never makes room on this PC, nor the reverse.
    if (this.deps.unloadPreviousOnLoad()) await this.unloadWhere((r) => hostKey(r.host) === key)
    this.checkDisposed()
    if (found.remote) return this.startRemote(model, cfg, host!)
    const exe = await this.deps.engine.resolveExe()
    this.checkDisposed()
    const port = await freePort()
    this.checkDisposed()
    const instanceId = randomUUID()
    const { loaded: adapters, notes: adapterNotes } = resolveAdapters(cfg.adapters ?? [], this.deps.getAdapter ?? (() => undefined))
    const args = buildServerArgs({ model, config: cfg, port, alias: model.name, adapters })
    const handle = startLocalServer({
      exe,
      args,
      port,
      spawnServer: this.deps.spawnServer,
      killProcess: this.deps.killProcess,
      unloadWaitMs: this.deps.unloadWaitMs
    })

    const info: LoadedInstance = {
      instanceId,
      modelId,
      modelName: model.name,
      port,
      pid: handle.pid,
      status: 'loading',
      progress: 0.02,
      stage: 'Starting',
      config: cfg,
      startedAt: Date.now(),
      contextLength: null,
      hasVision: false,
      adapters,
      estimate: this.deps.estimate(model, cfg)
    }
    const run = this.newRun(info, handle, [`$ llama-server ${args.join(' ')}`, ...adapterNotes])
    this.running.set(instanceId, run)
    this.attachHandle(run, handle)
    run.ready = this.waitReady(run)
    // Avoid an unhandled rejection when nobody awaits a failed background load.
    run.ready.catch(() => undefined)
    void this.savePids()
    this.emit()
    return { ready: run.ready, started: true }
  }

  private newRun(info: LoadedInstance, handle: ServerHandle | null, log: string[], host?: RemoteHost): Running {
    let reject!: (err: Error) => void
    const promise = new Promise<never>((_, rej) => (reject = rej))
    promise.catch(() => undefined)
    return {
      info,
      handle,
      host,
      log,
      ready: null!,
      exited: false,
      exitCode: null,
      users: 0,
      lastUsedAt: Date.now(),
      cancelled: false,
      cancel: { promise, reject }
    }
  }

  /** Registers a loading instance on a rental at once (so the window shows it and an unload can cancel it) and starts its server. */
  private startRemote(model: ModelEntry, cfg: LoadConfig, host: RemoteHost): { ready: Promise<LoadedInstance>; started: boolean } {
    const launcher = this.deps.launcher
    if (!launcher) throw new Error('Models on a vast.ai rental are not available in this build.')
    const instanceId = randomUUID()
    const { loaded: adapters, notes: adapterNotes } = resolveAdapters(cfg.adapters ?? [], this.deps.getAdapter ?? (() => undefined))
    const info: LoadedInstance = {
      instanceId,
      modelId: model.id,
      modelName: model.name,
      // The tunnel's port is only known once the server is started.
      port: 0,
      pid: null,
      status: 'loading',
      progress: 0.02,
      stage: 'Starting',
      config: cfg,
      startedAt: Date.now(),
      contextLength: null,
      hasVision: false,
      adapters,
      // The estimate would be about this PC's GPUs, not the rental's.
      estimate: null,
      host
    }
    const run = this.newRun(info, null, [`[remote] Starting ${model.name} on ${host.label}`, ...adapterNotes], host)
    this.running.set(instanceId, run)
    const note = (line: string): void => {
      run.log.push(line)
      if (run.log.length > MAX_LOG_LINES) run.log.splice(0, run.log.length - MAX_LOG_LINES)
    }
    run.ready = this.bringUpRemote(run, launcher, host, { model, config: cfg, alias: model.name, adapters, instanceId, note })
    run.ready.catch(() => undefined)
    this.watchReconnect()
    this.emit()
    return { ready: run.ready, started: true }
  }

  private async bringUpRemote(run: Running, launcher: RemoteLauncher, host: RemoteHost, launch: RemoteLaunch): Promise<LoadedInstance> {
    const started = launcher.start(host.rentalId, launch)
    // A server that comes up after the load was cancelled (or the app closed) must not be left running.
    started.then(
      (h) => {
        if (run.cancelled || this.disposed) void h.stop().catch(() => undefined)
      },
      () => undefined
    )
    let handle: ServerHandle
    try {
      handle = await Promise.race([started, run.cancel.promise])
    } catch (err) {
      if (run.cancelled) throw new Error('Load cancelled')
      // The rental could not start it (not running, not set up, ...): the sentence is the answer.
      this.onExit(run, null)
      throw err
    }
    if (run.cancelled || this.disposed) {
      void handle.stop().catch(() => undefined)
      throw new Error('Load cancelled')
    }
    run.handle = handle
    run.info.port = handle.port
    run.info.pid = handle.pid
    this.attachHandle(run, handle)
    this.emit()
    return this.waitReady(run)
  }

  /** Feeds a server's output into the log and the load progress, and reacts to its exit. */
  private attachHandle(run: Running, handle: ServerHandle): void {
    const splitter = new LogLineSplitter()
    let lastEmit = 0
    const onData = (chunk: string): void => {
      const { lines, dots } = splitter.push(chunk)
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
    handle.onOutput(onData)
    handle.onExit((exit) => {
      run.log.push(...splitter.flush())
      if (exit.reason) run.log.push(`[${exit.reason}]`)
      this.onExit(run, exit.code, exit.reason)
    })
  }

  private onExit(run: Running, code: number | null, reason?: string): void {
    if (run.exited) return
    run.exited = true
    run.exitCode = code
    run.exitReason = reason
    const wasReady = run.info.status === 'ready'
    const expected = run.info.status === 'stopping'
    this.running.delete(run.info.instanceId)
    void this.savePids()
    this.stopReconnectWatchIfIdle()
    this.emit()
    this.deps.onGone?.(run.info.instanceId)
    if (wasReady && !expected) {
      this.deps.onError({
        modelName: run.info.modelName,
        error: reason ?? `llama-server stopped unexpectedly (exit code ${code}).`,
        logTail: run.log.slice(-40)
      })
    }
  }

  private async waitReady(run: Running): Promise<LoadedInstance> {
    const remote = !!run.host
    const base = this.baseUrl(run.info)
    const deadline = Date.now() + (remote ? REMOTE_LOAD_TIMEOUT_MS : LOAD_TIMEOUT_MS)
    let healthy = false
    while (!healthy && Date.now() < deadline) {
      this.checkStillLoading(run)
      try {
        const res = await fetch(`${base}/health`, { signal: AbortSignal.timeout(remote ? REMOTE_HEALTH_TIMEOUT_MS : 2000) })
        healthy = res.ok
      } catch {
        // not listening yet
      }
      if (!healthy) await new Promise((r) => setTimeout(r, 400))
    }
    if (!healthy) {
      await this.unload(run.info.instanceId).catch(() => undefined)
      throw new Error('Timed out waiting for the model to load.')
    }

    try {
      const props = (await (await fetch(`${base}/props`, { signal: AbortSignal.timeout(remote ? REMOTE_HEALTH_TIMEOUT_MS : 5000) })).json()) as {
        n_ctx?: number
        default_generation_settings?: { n_ctx?: number }
        modalities?: { vision?: boolean }
      }
      run.info.contextLength = props.default_generation_settings?.n_ctx ?? props.n_ctx ?? null
      run.info.hasVision = !!props.modalities?.vision
    } catch {
      // props are informational only
    }
    await this.applyAdapterScales(run, base)
    // An unload (or a crash) while /props was being read: report that, not a ready model, or the
    // exit that follows would look like an unexpected stop.
    this.checkStillLoading(run)
    run.info.status = 'ready'
    run.info.progress = 1
    run.info.stage = 'Ready'
    run.lastUsedAt = Date.now()
    this.emit()
    return { ...run.info }
  }

  /** Sets the adapters' default strengths (they load at 1); a failure is logged and the load goes on. */
  private async applyAdapterScales(run: Running, base: string): Promise<void> {
    const adapters = run.info.adapters
    if (!adapters.some((a) => a.scale !== 1)) return
    try {
      const res = await fetch(`${base}/lora-adapters`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(adapters.map((a, id) => ({ id, scale: a.scale }))),
        signal: AbortSignal.timeout(10000)
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
    } catch (err) {
      run.log.push(`[adapter scales not set] ${(err as Error).message}; the adapters run at strength 1`)
      // Report what the server really does, so the UI, batch records and request defaults match.
      for (const a of run.info.adapters) a.scale = 1
    }
  }

  private checkStillLoading(run: Running): void {
    if (run.info.status === 'stopping') throw new Error('Load cancelled')
    if (run.exited) throw new Error(this.failureMessage(run))
  }

  private failureMessage(run: Running): string {
    if (run.exitReason) return `Could not load ${run.info.modelName}: ${run.exitReason}`
    const errLines = run.log.filter((l) => /error|failed|out of memory|unable|cannot|invalid/i.test(l)).slice(-6)
    const detail =
      describeExitCode(run.exitCode) ??
      describeLoadLog(run.log) ??
      (errLines.length ? errLines.join('\n') : run.log.slice(-6).join('\n'))
    return `llama-server exited while loading ${run.info.modelName} (code ${run.exitCode}).\n${detail}`
  }

  /**
   * Ends an instance. For a remote one this rejects when its server may still be running (the instance then stays
   * listed, ready again); it is never reported gone when it is not.
   */
  async unload(instanceId: string): Promise<void> {
    const run = this.running.get(instanceId)
    if (!run) return
    if (run.host) return this.unloadRemote(run)
    run.info.status = 'stopping'
    run.info.stage = 'Unloading'
    this.emit()
    await run.handle!.stop()
    if (!run.exited) this.onExit(run, null)
  }

  private unloadRemote(run: Running): Promise<void> {
    if (run.unloading) return run.unloading
    const p = (async () => {
      const before = { status: run.info.status, stage: run.info.stage }
      run.info.status = 'stopping'
      run.info.stage = 'Unloading'
      this.emit()
      if (!run.handle) {
        // Still being started: nothing to stop yet; a server that arrives later is stopped when it does.
        run.cancelled = true
        run.cancel.reject(new Error('Load cancelled'))
        this.onExit(run, null)
        return
      }
      try {
        await run.handle.stop()
      } catch (err) {
        if (!run.exited) {
          run.info.status = before.status
          run.info.stage = before.stage
          run.log.push(`[could not stop it: ${(err as Error).message}]`)
          this.emit()
          throw err
        }
      }
      if (!run.exited) this.onExit(run, null)
    })().finally(() => {
      run.unloading = undefined
    })
    run.unloading = p
    return p
  }

  /** Unloads the instances `pred` selects; every one is tried, and the first failure is thrown afterwards. */
  private async unloadWhere(pred: (run: Running) => boolean): Promise<void> {
    const ids = [...this.running.values()].filter(pred).map((r) => r.info.instanceId)
    const results = await Promise.allSettled(ids.map((id) => this.unload(id)))
    const failed = results.find((r): r is PromiseRejectedResult => r.status === 'rejected')
    if (failed) throw failed.reason
  }

  /** Unloads everything, on this PC and on rentals (the window's "eject all"). */
  async unloadAll(): Promise<void> {
    await this.unloadWhere(() => true)
  }

  /** Unloads what runs on this PC only; a rental's models stay (benchmarks and anything else that needs the GPUs here). */
  async unloadAllLocal(): Promise<void> {
    await this.unloadWhere((r) => !r.host)
  }

  /**
   * Synchronous end for app shutdown; loads that are still starting are refused from here on. A local server is
   * killed; a remote one keeps running on its rental and is only disconnected.
   */
  killAllSync(): void {
    this.disposed = true
    for (const run of this.running.values()) {
      run.info.status = 'stopping'
      run.cancelled = true
      if (run.handle) {
        try {
          run.handle.detachSync()
        } catch {
          // the others must still be handled
        }
      } else run.cancel.reject(new Error('Load cancelled'))
    }
    this.running.clear()
    if (this.reconnectTimer) clearInterval(this.reconnectTimer)
    this.reconnectTimer = null
    void this.savePids()
  }

  /** Kills llama-server processes left behind if the app crashed last time. */
  async cleanupOrphans(): Promise<void> {
    await killOrphanProcesses(this.deps.pidFile)
    await this.savePids()
  }

  private savePids(): Promise<void> {
    const pids = [...this.running.values()].map((r) => r.handle?.pid).filter((p): p is number => !!p)
    return savePidFile(this.deps.pidFile, pids)
  }

  /** While a remote instance exists, its tunnel is looked at every 2 s so the window can say "reconnecting". */
  private watchReconnect(): void {
    if (this.reconnectTimer || this.disposed) return
    this.reconnectTimer = setInterval(() => this.pollReconnect(), this.deps.reconnectPollMs ?? RECONNECT_POLL_MS)
    this.reconnectTimer.unref?.()
  }

  private pollReconnect(): void {
    let changed = false
    for (const run of this.running.values()) {
      if (!run.handle || !run.host) continue
      const now = run.handle.reconnecting()
      if (now === !!run.info.reconnecting) continue
      if (now) run.info.reconnecting = true
      else delete run.info.reconnecting
      changed = true
    }
    if (changed) this.emit()
    this.stopReconnectWatchIfIdle()
  }

  private stopReconnectWatchIfIdle(): void {
    if (!this.reconnectTimer) return
    if ([...this.running.values()].some((r) => r.host)) return
    clearInterval(this.reconnectTimer)
    this.reconnectTimer = null
  }

  private emit(): void {
    this.deps.onChange(this.list())
  }
}
