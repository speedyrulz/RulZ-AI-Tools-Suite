// llama-server on a rental, as a ServerHandle for InstanceManager (docs/vast-design.md, section 7): started detached
// on the instance, reached through a local ssh tunnel that is respawned when it drops, its log polled by byte offset.
import { createHash } from 'node:crypto'
import { connect } from 'node:net'
import type { RemoteGpu, RemoteHost, RemoteModelDef } from '@shared/remote'
import type { AdapterEntry, InstanceAdapter, LoadConfig, MemoryEstimate, ModelEntry, SidecarFile } from '@shared/types'
import { describeLoadLog, freePort } from '../engine/process'
import type { RemoteLaunch, RemoteLauncher, ServerExit, ServerHandle } from '../engine/serverHandle'
import type { RentalDirectory, RentalInfo } from './contract'
import { fileList, recordedComplete, slugOf, type RemoteFile } from './models'
import { buildRemoteArgs, type RemotePaths } from './remoteArgs'
import { launchCheckScript, parseComplete, parseGpus, PORT_PLACEHOLDER, serverLogScript, serverStartScript, serverStatusScript, serverStopScript, type CompleteEntry } from './scripts'
import { marker, markers, SshError, type RemoteSsh, type TunnelProc } from './ssh'
import { cachedHash, localSource, sharedHashCache, type HashCache } from './transfer'

export interface RemoteLauncherDeps {
  ssh: RemoteSsh
  directory: RentalDirectory
  getDef: (modelId: string) => RemoteModelDef | undefined
  getModel: (id: string) => ModelEntry | undefined
  getAdapter?: (id: string) => AdapterEntry | undefined
  /**
   * The model's default load settings and those of its modes: the files were uploaded for all of them under the names
   * of that full list, so a load finds its own draft and adapters there. Without them a load's files are named alone.
   */
  getDefaults?: (modelId: string) => LoadConfig | undefined
  getModeLoads?: (modelId: string) => LoadConfig[]
  /** The draft this load uses (InstanceManager's own choice); undefined = none. */
  draftFor: (model: ModelEntry, cfg: LoadConfig) => SidecarFile | undefined
  /** Memory estimate for the rental's GPUs; null = unknown. */
  estimate?: (model: ModelEntry, cfg: LoadConfig, gpus: RemoteGpu[]) => MemoryEstimate | null
  /** A free local port for the tunnel. Default: the OS picks one. */
  pickPort?: () => Promise<number>
  /** Whether something accepts connections on a local port. Default: a TCP connect. */
  portOpen?: (port: number) => Promise<boolean>
  /** Local file hashes the uploads recorded; a changed local file counts as not on the instance. Default: the shared cache. */
  hashCache?: HashCache
}

/** How long the tunnel may stay down before the server is looked at over a new route. */
export const TUNNEL_GIVE_UP_MS = 120_000
const LOADING_POLL_MS = 1000
const LOADED_POLL_MS = 10_000
const MAX_BACKOFF_MS = 15_000
const MAX_BUFFER_CHARS = 1_000_000
const LOADED_RE = /(server is listening|all slots are idle|starting the main loop)/i

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))

function tcpOpen(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const s = connect({ host: '127.0.0.1', port })
    const done = (ok: boolean): void => {
      s.destroy()
      resolve(ok)
    }
    s.setTimeout(1000, () => done(false))
    s.once('connect', () => done(true))
    s.once('error', () => done(false))
  })
}

const isTransport = (err: unknown): boolean => err instanceof SshError && (err.kind === 'unreachable' || err.kind === 'transport' || err.kind === 'timeout')

/** The hash that decides whether a running server can be reused: arguments, engine build, and the files with their sizes and checksums. */
export function serverHash(tag: string, args: string[], files: CompleteEntry[]): string {
  const h = createHash('sha256')
  h.update(JSON.stringify([tag, args, [...files].sort((a, b) => a.name.localeCompare(b.name)).map((f) => [f.name, f.size, f.sha256, f.source])]))
  return h.digest('hex')
}

interface CheckResult {
  root: string
  engineOk: boolean
  complete: CompleteEntry[]
  gpus: RemoteGpu[]
  nvidiaSmi: boolean
  foreign: boolean
  /** Real sizes of the files in the model folder. */
  files: Map<string, number>
  /** Servers of this engine that run on the instance, whoever started them. */
  runs: Array<{ id: string; port: number; hash: string }>
}

/** Run ids of the handles of this process that have not ended, per rental. */
type Registry = Map<number, Set<string>>

export function createRemoteLauncher(deps: RemoteLauncherDeps): RemoteLauncher {
  const live: Registry = new Map()
  // One start at a time per rental: what one finds running on the instance must not change under it.
  const queues = new Map<number, Promise<unknown>>()
  return {
    start: (rentalId, launch) => {
      const run = (queues.get(rentalId) ?? Promise.resolve()).then(() => startRemote(deps, live, rentalId, launch))
      queues.set(rentalId, run.catch(() => undefined))
      return run
    }
  }
}

async function startRemote(deps: RemoteLauncherDeps, live: Registry, rentalId: number, launch: RemoteLaunch): Promise<ServerHandle> {
  const info = deps.directory.info(rentalId)
  if (!info) throw new Error('This rental is not in the list any more.')
  if (!info.running) throw new Error(`${info.label} is not running. Start it on the vast.ai page first.`)
  const def = deps.getDef(launch.model.id)
  if (!def) throw new Error(`The remote model "${launch.model.name}" is not saved any more.`)
  const slug = slugOf(def.id)
  const files = fileList(def, { getModel: deps.getModel, getAdapter: deps.getAdapter, config: deps.getDefaults?.(def.id), extra: deps.getModeLoads?.(def.id), for: launch.config })
  const tag = info.engineTag
  const host: RemoteHost = { kind: 'vast', rentalId, label: info.label }

  launch.note(`Checking ${info.label}...`)
  let check = await runCheck(deps, rentalId, tag, slug)
  if (!check.engineOk) throw new Error(`The engine ${tag} is not installed on ${info.label}. Set the rental up first.`)
  const missing = missingFiles(files, check, deps.hashCache ?? sharedHashCache)
  if (missing.length) {
    throw new Error(`The files of ${launch.model.name} are not complete on ${info.label} (${missing.slice(0, 3).join(', ')}${missing.length > 3 ? ', ...' : ''}). Set the model up on this rental first.`)
  }

  const base = `${check.root}/models/${slug}`
  const paths = remotePaths(base, files, launch.adapters)
  const draft = deps.draftFor(launch.model, launch.config)
  const args = buildRemoteArgs({ model: launch.model, config: launch.config, port: PORT_PLACEHOLDER, alias: launch.alias, paths, draft, gpus: check.gpus.map((g) => ({ index: g.index, freeBytes: Math.max(0, g.totalMiB - g.usedMiB) * 1048576 })) })
  const used = files.map((f) => check.complete.find((c) => c.name === f.name) as CompleteEntry)
  const hash = serverHash(tag, args, used)

  // Every load has a new run id, so servers left behind (a lost connection, a closed app) are found by what runs on the
  // instance: one with the same hash that no live handle of this process owns is taken over, the others are stopped.
  const owned = live.get(rentalId) ?? new Set<string>()
  live.set(rentalId, owned)
  const orphans = check.runs.filter((r) => !owned.has(r.id))
  const adopt = orphans.find((r) => r.hash === hash && r.port > 0)
  const others = orphans.filter((r) => r !== adopt)
  const runId = adopt?.id ?? launch.instanceId
  owned.add(runId)
  try {
    for (const r of others) {
      launch.note(`Stopping an older server on ${info.label}...`)
      await stopRemote(deps, rentalId, r.id)
    }
    let remotePort: number
    if (adopt) {
      launch.note('A server of this model that was left running on the instance is used again.')
      remotePort = adopt.port
    } else {
      if (others.length) check = await runCheck(deps, rentalId, tag, slug)
      preflight(deps, info, launch, check)
      launch.note(`Starting llama-server on ${info.label}...`)
      const est = deps.estimate?.(launch.model, launch.config, check.gpus) ?? null
      const res = await deps.ssh.run(
        rentalId,
        serverStartScript({ id: runId, tag, args, hash, expectMiB: est ? est.gpuBytes / 1048576 : null }),
        { timeoutMs: 90_000 }
      )
      remotePort = Number(marker(res.stdout, 'PORT'))
      if (res.code !== 0 || marker(res.stdout, 'STARTED') !== '1' || !Number.isInteger(remotePort)) {
        const tail = Buffer.from(marker(res.stdout, 'LOG') ?? '', 'base64').toString('utf8').trim()
        const known = describeLoadLog(tail.split(/\r?\n/), { label: info.label, tag })
        if (known) throw new Error(known)
        throw new Error(`${marker(res.stdout, 'ERR') ?? 'llama-server could not be started on the instance.'}${tail ? `\n${tail}` : ''}`)
      }
    }
    const localPort = await (deps.pickPort ?? freePort)()
    const handle = new RemoteHandle(deps, host, rentalId, tag, runId, localPort, remotePort, () => owned.delete(runId))
    handle.begin()
    return handle
  } catch (err) {
    // No handle owns it: a server that came up anyway is found, and taken over or stopped, by the next load.
    owned.delete(runId)
    throw err
  }
}

async function runCheck(deps: RemoteLauncherDeps, rentalId: number, tag: string, slug: string): Promise<CheckResult> {
  const res = await deps.ssh.run(rentalId, launchCheckScript({ id: 'check', tag, slug }), { timeoutMs: 45_000 })
  const out = res.stdout
  return {
    root: marker(out, 'ROOT') || '/workspace/rulz-llm',
    engineOk: marker(out, 'ENGINE_OK') === '1',
    complete: parseComplete(Buffer.from(marker(out, 'COMPLETE_B64') ?? '', 'base64').toString('utf8')),
    gpus: parseGpus(out),
    nvidiaSmi: marker(out, 'NVSMI') === '1',
    foreign: marker(out, 'FOREIGN') === '1',
    files: new Map(markers(out, 'FILE').map((v) => [v.slice(0, v.lastIndexOf('|')), Number(v.slice(v.lastIndexOf('|') + 1))] as [string, number])),
    runs: markers(out, 'RUN').map((v) => {
      const [id, port, hash] = v.split('|')
      return { id, port: Number(port) || 0, hash: hash ?? '' }
    })
  }
}

/** Names of required files that are not complete: not recorded, not on disk with the recorded size, from another source, or changed on this PC. */
function missingFiles(files: RemoteFile[], check: CheckResult, cache: HashCache): string[] {
  const by = new Map(check.complete.map((c) => [c.name, c]))
  return files
    .filter((f) => !recordedComplete(f, by.get(f.name), check.files.get(f.name), f.localPath ? { sha256: cachedHash(f.localPath, cache), source: localSource(f.localPath) } : undefined))
    .map((f) => f.name)
}

function remotePaths(base: string, files: RemoteFile[], adapters: InstanceAdapter[]): RemotePaths {
  const at = (f: RemoteFile): string => `${base}/${f.name}`
  const main = files.find((f) => f.role === 'main')
  if (!main) throw new Error('The model has no main file.')
  const adapterFiles = files.filter((f) => f.role === 'adapter')
  const paths: RemotePaths = {
    main: at(main),
    mmproj: files.find((f) => f.role === 'mmproj') ? at(files.find((f) => f.role === 'mmproj') as RemoteFile) : undefined,
    draft: files.find((f) => f.role === 'draft') ? at(files.find((f) => f.role === 'draft') as RemoteFile) : undefined,
    adapters: []
  }
  for (const a of adapters) {
    const f = adapterFiles.find((x) => x.localPath === a.path)
    if (!f) throw new Error(`The adapter ${a.name} is not on the instance. Set the model up there again with this adapter selected.`)
    paths.adapters.push(at(f))
  }
  return paths
}

/** Refuses on an attached instance, notes on one the app rented (design section 7). */
function preflight(deps: RemoteLauncherDeps, info: RentalInfo, launch: RemoteLaunch, check: CheckResult): void {
  const problems: string[] = []
  if (check.foreign && check.nvidiaSmi) problems.push(`Another program seems to be using a GPU on ${info.label} (a training job, for example).`)
  const est = deps.estimate?.(launch.model, launch.config, check.gpus) ?? null
  if (est && check.gpus.length) {
    const free = check.gpus.reduce((a, g) => a + Math.max(0, g.totalMiB - g.usedMiB), 0) * 1048576
    if (est.gpuBytes > free) problems.push(`The model needs about ${(est.gpuBytes / 2 ** 30).toFixed(1)} GiB of GPU memory and ${info.label} has ${(free / 2 ** 30).toFixed(1)} GiB free.`)
  }
  if (!problems.length) return
  if (info.origin === 'attached') throw new Error(`${problems.join(' ')} The instance was not rented by this app, so nothing was started.`)
  for (const p of problems) launch.note(`Warning: ${p}`)
}

/** Stops the server of one instance id; throws a sentence unless it is confirmed gone. */
async function stopRemote(deps: RemoteLauncherDeps, rentalId: number, id: string): Promise<void> {
  let last: unknown
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await deps.ssh.run(rentalId, serverStopScript(id), { timeoutMs: 90_000 })
      if (res.code === 0 && marker(res.stdout, 'STOPPED') === '1') return
      throw new Error(marker(res.stdout, 'ERR') ?? 'llama-server did not stop.')
    } catch (err) {
      last = err
      if (!isTransport(err)) break
      await sleep(2000 * (attempt + 1))
    }
  }
  const why = last instanceof Error ? last.message : 'unknown reason'
  throw new Error(`Could not confirm that the server stopped: ${why} It may still be running on the instance.`)
}

class RemoteHandle implements ServerHandle {
  readonly pid = null
  readonly port: number

  private readonly outputFns: Array<(chunk: string) => void> = []
  private readonly exitFns: Array<(exit: ServerExit) => void> = []
  private buffer: string[] = []
  private bufferChars = 0
  private exited: ServerExit | null = null
  /** The server is known to be gone. A transport exit is not: stop() still has to ask the instance. */
  private confirmed = false
  private closed = false
  private stopping = false

  private tunnel: TunnelProc | null = null
  private up = false
  private gen = 0
  private failures = 0
  private lastOkAt = Date.now()
  private verifying = false
  private aliveVerifications = 0
  private respawn: NodeJS.Timeout | null = null
  private logTimer: NodeJS.Timeout | null = null
  private logOffset = 0
  private loaded = false
  private decoder = new TextDecoder('utf-8')

  constructor(
    private readonly deps: RemoteLauncherDeps,
    readonly host: RemoteHost,
    private readonly rentalId: number,
    /** The llama.cpp build the server runs. */
    private readonly tag: string,
    private readonly id: string,
    localPort: number,
    private readonly remotePort: number,
    /** Takes this handle out of the launcher's list of live handles. */
    private readonly release: () => void
  ) {
    this.port = localPort
  }

  /** Starts the tunnel and the log polling. */
  begin(): void {
    this.spawnTunnel()
    this.scheduleLog(0)
  }

  onOutput(fn: (chunk: string) => void): void {
    for (const c of this.buffer) fn(c)
    this.outputFns.push(fn)
  }

  onExit(fn: (exit: ServerExit) => void): void {
    if (this.exited) {
      fn(this.exited)
      return
    }
    this.exitFns.push(fn)
  }

  reconnecting(): boolean {
    return !this.exited && !this.closed && !this.up
  }

  detachSync(): void {
    this.closed = true
    this.clearTimers()
    this.killTunnel()
    this.release()
  }

  async stop(): Promise<void> {
    // Only a confirmed exit makes this a no-op: after the tunnel gave up the server may still run.
    if (this.exited && this.confirmed) return
    this.stopping = true
    try {
      await stopRemote(this.deps, this.rentalId, this.id)
    } catch (err) {
      this.stopping = false
      throw err
    }
    this.confirmed = true
    if (this.exited) return
    // The last lines of the log, best effort, then the tunnel goes.
    await this.pollOnce().catch(() => undefined)
    this.finish({ code: 0 }, true)
  }

  // ---------- Exit ----------

  private finish(exit: ServerExit, confirmed: boolean): void {
    if (this.exited) return
    this.exited = exit
    this.confirmed = confirmed
    this.closed = true
    this.release()
    this.clearTimers()
    this.killTunnel()
    for (const fn of this.exitFns.splice(0)) {
      try {
        fn(exit)
      } catch {
        // a listener's problem
      }
    }
  }

  private clearTimers(): void {
    if (this.respawn) clearTimeout(this.respawn)
    if (this.logTimer) clearTimeout(this.logTimer)
    this.respawn = null
    this.logTimer = null
  }

  private killTunnel(): void {
    this.gen++
    this.up = false
    const t = this.tunnel
    this.tunnel = null
    t?.kill()
  }

  // ---------- Output ----------

  private emit(text: string): void {
    if (!text) return
    this.buffer.push(text)
    this.bufferChars += text.length
    while (this.bufferChars > MAX_BUFFER_CHARS && this.buffer.length > 1) this.bufferChars -= (this.buffer.shift() as string).length
    if (!this.loaded && LOADED_RE.test(text)) this.loaded = true
    for (const fn of this.outputFns) {
      try {
        fn(text)
      } catch {
        // a listener's problem
      }
    }
  }

  private tailText(lines = 12): string {
    return this.buffer.join('').split(/\r?\n/).filter((l) => l.trim()).slice(-lines).join('\n')
  }

  /** Why the server ended: a plain explanation of a load failure, else `what` and the tail of its log. */
  private endReason(what: string): string {
    const known = this.loaded ? undefined : describeLoadLog(this.buffer.join('').split(/\r?\n/), { label: this.host.label, tag: this.tag })
    if (known) return known
    const tail = this.tailText()
    return `${what}${tail ? `\nLast lines of its log:\n${tail}` : ''}`
  }

  // ---------- Tunnel ----------

  private spawnTunnel(): void {
    if (this.closed) return
    this.respawn = null
    let proc: TunnelProc
    try {
      proc = this.deps.ssh.tunnel(this.rentalId, this.port, this.remotePort)
    } catch (err) {
      this.tunnelDown((err as Error).message)
      return
    }
    const gen = ++this.gen
    this.tunnel = proc
    this.up = false
    void proc.exited.then((r) => {
      if (gen !== this.gen) return
      this.tunnel = null
      this.tunnelDown(r.stderr.trim().split(/\r?\n/).pop() || `ssh ended (code ${r.code})`)
    })
    void this.waitUp(gen, proc)
  }

  private async waitUp(gen: number, proc: TunnelProc): Promise<void> {
    const open = this.deps.portOpen ?? tcpOpen
    for (let i = 0; i < 40 && gen === this.gen && !this.closed; i++) {
      if (await open(this.port).catch(() => false)) {
        if (gen !== this.gen || this.closed) return
        this.up = true
        this.failures = 0
        this.aliveVerifications = 0
        this.lastOkAt = Date.now()
        return
      }
      await sleep(250)
    }
    // Never answered: end this attempt so the normal retry takes over.
    if (gen === this.gen && !this.closed && !this.up) proc.kill()
  }

  private tunnelDown(why: string): void {
    if (this.closed) return
    if (this.up) this.lastOkAt = Date.now()
    this.up = false
    this.failures++
    if (this.failures >= 2) this.deps.ssh.routeFailed(this.rentalId)
    if (Date.now() - this.lastOkAt >= TUNNEL_GIVE_UP_MS && !this.verifying) {
      void this.verifyServer(why)
      return
    }
    const delay = Math.min(MAX_BACKOFF_MS, 1000 * 2 ** Math.min(this.failures - 1, 4))
    this.respawn = setTimeout(() => this.spawnTunnel(), delay)
  }

  /** The tunnel has been down for too long: ask the instance over a fresh route whether the server still runs. */
  private async verifyServer(why: string): Promise<void> {
    this.verifying = true
    try {
      this.deps.ssh.routeFailed(this.rentalId)
      let alive = false
      try {
        const res = await this.deps.ssh.run(this.rentalId, serverStatusScript(this.id), { timeoutMs: 45_000 })
        alive = marker(res.stdout, 'ALIVE') === '1'
      } catch (err) {
        this.finish({ code: null, reason: `The connection to ${this.host.label} was lost and could not be re-made for 2 minutes (${(err as Error).message}).` }, false)
        return
      }
      if (this.closed) return
      if (!alive) {
        await this.pollOnce().catch(() => undefined)
        this.finish({ code: null, reason: this.endReason(`llama-server on ${this.host.label} is no longer running.`) }, true)
        return
      }
      this.aliveVerifications++
      if (this.aliveVerifications >= 2) {
        this.finish({ code: null, reason: `The server still runs on ${this.host.label}, but this PC cannot make a tunnel to it (${why}).` }, false)
        return
      }
      this.lastOkAt = Date.now()
      this.failures = 0
      this.spawnTunnel()
    } finally {
      this.verifying = false
    }
  }

  // ---------- Log ----------

  private scheduleLog(delay: number): void {
    if (this.closed) return
    this.logTimer = setTimeout(() => {
      this.logTimer = null
      void this.pollOnce()
        .catch(() => undefined)
        .finally(() => this.scheduleLog(this.loaded ? LOADED_POLL_MS : LOADING_POLL_MS))
    }, delay)
  }

  /** Reads what the log gained since the last read; a dead process ends the handle with the log's tail. */
  private async pollOnce(): Promise<void> {
    for (let round = 0; round < 20; round++) {
      const res = await this.deps.ssh.run(this.rentalId, serverLogScript(this.id, this.logOffset), { timeoutMs: 30_000 })
      const out = res.stdout
      if (marker(out, 'LOG_RESET') === '1') {
        this.decoder = new TextDecoder('utf-8')
        this.logOffset = 0
      }
      const bytes = Buffer.from(marker(out, 'LOG') ?? '', 'base64')
      const next = Number(marker(out, 'LOG_NEXT'))
      if (bytes.length) this.emit(this.decoder.decode(bytes, { stream: true }))
      if (Number.isFinite(next)) this.logOffset = next
      if (bytes.length >= 65536) continue
      if (marker(out, 'ALIVE') === '0' && !this.stopping && !this.closed) {
        this.finish({ code: null, reason: this.endReason(`llama-server stopped on ${this.host.label}.`) }, true)
      }
      return
    }
  }
}
