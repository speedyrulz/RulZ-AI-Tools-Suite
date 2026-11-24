// The ssh side of a rental as the rental manager uses it (RentalOps, contract.ts): probe, watchdog, engine,
// model files, keepalive. Everything it talks to is injected, so tests need neither ssh nor the network.
import type { AutoStop, RemoteModelDef } from '@shared/remote'
import { REMOTE_MODEL_PREFIX } from '@shared/remote'
import type { AdapterEntry, LoadConfig, ModelEntry } from '@shared/types'
import type { ProbeResult, RentalDirectory, RentalOps, Secrets, TransferProgress } from './contract'
import { fileList, recordedComplete, slugOf, type FileListDeps, type RemoteFile } from './models'
import {
  armScript,
  disarmScript,
  engineInstallScript,
  hfConfigLine,
  hfDownloadScript,
  hfStopScript,
  keepaliveLoopScript,
  markCompleteScript,
  modelStateScript,
  onstartScript,
  parseComplete,
  parseProbe,
  probeScript,
  removeModelScript,
  touchScript,
  type EngineInstallInput
} from './scripts'
import { marker, markers, type RunResult, type ScriptRunner, type SshProcess } from './ssh'
import { cachedHash, localSource, sharedHashCache, uploadFile, type HashCache } from './transfer'

export interface OpsSsh extends ScriptRunner {
  /** A process with a held stdin; the keepalive loop. */
  open(rentalId: number, script: string): SshProcess
  routeFailed?(rentalId: number): void
}

export interface OpsDeps {
  directory: RentalDirectory
  secrets: Secrets
  ssh: OpsSsh
  models: { get(id: string): RemoteModelDef | undefined }
  getModel: (id: string) => ModelEntry | undefined
  getAdapter?: (id: string) => AdapterEntry | undefined
  /** The model's load settings (its adapters go with it). */
  getDefaults?: (id: string) => LoadConfig | undefined
  /** The load settings of the model's modes (their adapters and draft go with it too); none = no mode has its own. */
  getModeLoads?: (id: string) => LoadConfig[]
  /** Default: the global fetch. */
  fetch?: typeof fetch
  hashCache?: HashCache
}

const KEEPALIVE_MS = 60_000
const MAX_LOG = 400
/** How often a gated download asks for a fresh signed address. */
const GATED_TRIES = 12

const parseUrl = (u: string, base: string): URL | null => (URL.canParse(u, base) ? new URL(u, base) : null)
const isHfHost = (u: URL): boolean => u.protocol === 'https:' && u.hostname === 'huggingface.co' && u.port === ''

const LLAMA_REPO = 'ggml-org/llama.cpp'

/** The moving rate over the last `windowMs` of the byte counts it is given: "840" Mbps, or null until two seconds are covered. */
export class RateMeter {
  private readonly samples: Array<{ at: number; bytes: number }> = []

  constructor(
    private readonly now: () => number = Date.now,
    private readonly windowMs = 10_000
  ) {}

  add(bytes: number): void {
    const at = this.now()
    this.samples.push({ at, bytes })
    while (this.samples.length > 2 && at - this.samples[1].at >= this.windowMs) this.samples.shift()
  }

  /** Megabits per second, rounded. */
  mbps(): number | null {
    const first = this.samples[0]
    const last = this.samples[this.samples.length - 1]
    if (!first || last.at - first.at < 2000) return null
    return Math.max(0, Math.round(((last.bytes - first.bytes) * 8) / ((last.at - first.at) / 1000) / 1e6))
  }
}

export const llamaAssetNames = (tag: string): { llama: string; cudart: string } => ({
  llama: `llama-${tag}-bin-ubuntu-cuda-12.8-x64.tar.gz`,
  cudart: `cudart-llama-${tag}-bin-ubuntu-cuda-12.8-x64.tar.gz`
})

interface KeepAlive {
  active: boolean
  proc: SshProcess | null
  ticker: NodeJS.Timeout | null
  retry: NodeJS.Timeout | null
  failures: number
}

export class SshRentalOps implements RentalOps {
  private readonly fetchFn: typeof fetch
  private readonly cache: HashCache
  private readonly logs = new Map<number, string[]>()
  private readonly keeps = new Map<number, KeepAlive>()
  private readonly running = new Map<number, Set<AbortController>>()
  private readonly digests = new Map<string, EngineInstallInput>()

  constructor(private readonly deps: OpsDeps) {
    this.fetchFn = deps.fetch ?? ((...a) => fetch(...a))
    this.cache = deps.hashCache ?? sharedHashCache
  }

  // ---------- Log and bookkeeping ----------

  private note(rentalId: number, line: string): void {
    const clean = this.deps.secrets.scrub(line)
    const buf = this.logs.get(rentalId) ?? []
    buf.push(clean)
    if (buf.length > MAX_LOG) buf.splice(0, buf.length - MAX_LOG)
    this.logs.set(rentalId, buf)
    this.deps.directory.note(rentalId, clean)
  }

  log(rentalId: number): string[] {
    return [...(this.logs.get(rentalId) ?? [])]
  }

  /** A controller that ends when `signal` does or when the rental is disconnected. */
  private track(rentalId: number, signal?: AbortSignal): { ctl: AbortController; done: () => void } {
    const ctl = new AbortController()
    const set = this.running.get(rentalId) ?? new Set()
    set.add(ctl)
    this.running.set(rentalId, set)
    const onAbort = (): void => ctl.abort()
    if (signal?.aborted) ctl.abort()
    else signal?.addEventListener('abort', onAbort, { once: true })
    return {
      ctl,
      done: () => {
        signal?.removeEventListener('abort', onAbort)
        set.delete(ctl)
      }
    }
  }

  private failure(res: RunResult, fallback: string): Error {
    return new Error(marker(res.stdout, 'ERR') ?? (res.stderr.trim().split(/\r?\n/).pop() || fallback))
  }

  // ---------- RentalOps ----------

  async probe(rentalId: number, signal?: AbortSignal): Promise<ProbeResult> {
    const res = await this.deps.ssh.run(rentalId, probeScript(), { timeoutMs: 45_000, signal })
    const p = parseProbe(res.stdout)
    const state = p.watchdogState
    const armed = p.watchdogRunning && state === 'ok'
    return {
      gpus: p.gpus,
      diskFreeBytes: p.diskFreeBytes,
      // No GPU tool means no GPU process, not a foreign one.
      foreignProcesses: p.foreign && p.nvidiaSmi,
      engines: p.engines,
      watchdog: { armed, ...(state.startsWith('error:') ? { error: state.slice(6).trim() } : !armed && state === 'ok' ? { error: 'The watchdog is not running.' } : {}) },
      models: p.modelSlugs.map((s) => REMOTE_MODEL_PREFIX + s)
    }
  }

  async armWatchdog(
    rentalId: number,
    limits: { autoStop: AutoStop; capAt: number | null; origin: 'app' | 'attached' }
  ): Promise<{ armed: boolean; error?: string }> {
    const idleMinutes = limits.autoStop.enabled ? Math.max(1, Math.round(limits.autoStop.idleMinutes)) : 0
    // The manager's times are epoch milliseconds; the watchdog compares epoch seconds.
    const capSeconds = limits.capAt === null ? null : Math.round(limits.capAt > 1e11 ? limits.capAt / 1000 : limits.capAt)
    if (idleMinutes === 0 && capSeconds === null) {
      await this.deps.ssh.run(rentalId, disarmScript(), { timeoutMs: 30_000 })
      return { armed: false }
    }
    const res = await this.deps.ssh.run(rentalId, armScript({ idleMinutes, capAt: capSeconds, origin: limits.origin, capHours: limits.autoStop.capHours === null ? 0 : Math.max(1, Math.round(limits.autoStop.capHours)) }), { timeoutMs: 60_000 })
    const armed = marker(res.stdout, 'ARMED') === '1'
    if (armed) return { armed: true }
    const error = marker(res.stdout, 'ERR') ?? 'The watchdog could not be armed.'
    this.note(rentalId, `Watchdog: ${error}`)
    return { armed: false, error }
  }

  onstartScript(limits: { autoStop: AutoStop }): string {
    const a = limits.autoStop
    return onstartScript({ idleMinutes: a.enabled ? Math.max(1, Math.round(a.idleMinutes)) : 0, capHours: a.capHours === null ? null : Math.max(1, Math.round(a.capHours)) })
  }

  // ---------- Engine ----------

  /** Both packages of a llama.cpp release with the sha256 digests the release API reports. */
  async releaseDigests(tag: string): Promise<EngineInstallInput> {
    const cached = this.digests.get(tag)
    if (cached) return cached
    let res: Response
    try {
      res = await this.fetchFn(`https://api.github.com/repos/${LLAMA_REPO}/releases/tags/${encodeURIComponent(tag)}`, {
        headers: { Accept: 'application/vnd.github+json', 'User-Agent': 'rulz-llm-suite', 'X-GitHub-Api-Version': '2022-11-28' }
      })
    } catch (err) {
      throw new Error(`Could not reach GitHub to look up llama.cpp ${tag}: ${(err as Error).message}`)
    }
    if (res.status === 404) throw new Error(`llama.cpp has no release ${tag}. Set another build under Settings on the vast.ai page.`)
    if (res.status === 403 || res.status === 429) throw new Error('GitHub refused the lookup of the llama.cpp release (rate limit). Try again in a few minutes.')
    if (!res.ok) throw new Error(`GitHub answered ${res.status} for the llama.cpp release ${tag}.`)
    const body = (await res.json()) as { assets?: Array<{ name?: string; digest?: string | null; size?: number }> }
    const names = llamaAssetNames(tag)
    const pick = (name: string): { url: string; sha256: string; size: number } => {
      const a = body.assets?.find((x) => x.name === name)
      if (!a) throw new Error(`llama.cpp ${tag} has no package ${name}, so it cannot be installed on a rental. Set another build tag.`)
      const m = /^sha256:([0-9a-f]{64})$/.exec(a.digest ?? '')
      if (!m) throw new Error(`GitHub gives no sha256 digest for ${name}, so the download could not be checked. Set another build tag.`)
      return { url: `https://github.com/${LLAMA_REPO}/releases/download/${encodeURIComponent(tag)}/${name}`, sha256: m[1], size: Number.isSafeInteger(a.size) ? (a.size as number) : 0 }
    }
    const input: EngineInstallInput = { tag, llama: pick(names.llama), cudart: pick(names.cudart) }
    this.digests.set(tag, input)
    return input
  }

  async installEngine(rentalId: number, tag: string, onProgress: (p: TransferProgress) => void, signal: AbortSignal): Promise<void> {
    const end = this.deps.directory.use(rentalId, 'setup', 'window')
    const t = this.track(rentalId, signal)
    try {
      onProgress({ stage: 'Looking up the llama.cpp release', bytesDone: 0, bytesTotal: 0 })
      const input = await this.releaseDigests(tag)
      const total = input.llama.size + input.cudart.size
      const done = new Map<string, number>()
      let stage = 'Installing the engine'
      const report = (): void => onProgress({ stage, bytesDone: [...done.values()].reduce((a, b) => a + b, 0), bytesTotal: total })
      report()
      const res = await this.deps.ssh.run(rentalId, engineInstallScript(input), {
        timeoutMs: 45 * 60_000,
        signal: t.ctl.signal,
        onLine: (line) => {
          const m = /^@rz (STAGE|PROG|VERSION|DEVICES)=(.*)$/.exec(line)
          if (!m) return
          if (m[1] === 'STAGE') stage = m[2]
          else if (m[1] === 'PROG') {
            const [label, d] = m[2].split('|')
            done.set(label, Number(d) || 0)
          } else this.note(rentalId, `Engine ${m[1].toLowerCase()}: ${m[2]}`)
          report()
        }
      })
      if (res.code !== 0 || marker(res.stdout, 'ENGINE_OK') !== tag) throw this.failure(res, 'The engine could not be installed.')
      onProgress({ stage: 'Engine ready', bytesDone: total, bytesTotal: total })
    } finally {
      t.done()
      end()
    }
  }

  // ---------- Models ----------

  private filesOf(def: RemoteModelDef): RemoteFile[] {
    const d: FileListDeps = { getModel: this.deps.getModel, getAdapter: this.deps.getAdapter, config: this.deps.getDefaults?.(def.id), extra: this.deps.getModeLoads?.(def.id) }
    return fileList(def, d)
  }

  /**
   * Size and sha256 of a Hugging Face file from HEAD requests by this PC. Asked without the token first; only a file
   * that is refused that way is asked with it. Such a gated file also comes with `signedUrl`, the short-lived address
   * Hugging Face redirects to: the instance downloads from that, so the token never leaves this PC ('' = there is no
   * such address). Redirects within huggingface.co (a renamed repository) are followed; no other host sees the token.
   */
  async headFile(url: string, token: string | null): Promise<{ size: number | null; sha256: string | null; signedUrl?: string }> {
    let res: Response
    let gated = false
    try {
      res = await this.hfHead(url, null)
      if (token && (res.status === 401 || res.status === 403 || res.status === 404)) {
        res = await this.hfHead(url, token)
        gated = true
      }
    } catch {
      return { size: null, sha256: null }
    }
    if (res.status === 401 || res.status === 403) throw new Error('Hugging Face refused access to this file. A gated or private repository needs a token with access: add it under Settings on the vast.ai page.')
    if (res.status === 404) throw new Error('Hugging Face has no such file. Check the repository, the file name and the revision.')
    const num = (v: string | null): number | null => (v && /^\d+$/.test(v) ? Number(v) : null)
    let size = num(res.headers.get('x-linked-size')) ?? num(res.headers.get('content-length'))
    const etag = (res.headers.get('x-linked-etag') ?? '').replace(/^W\//, '').replace(/"/g, '')
    const sha256 = /^[0-9a-f]{64}$/.test(etag) ? etag : null
    const loc = res.headers.get('location')
    // An error page's length is not the file's.
    if (res.status >= 400) return { size: null, sha256: null }
    if (size === null && res.status >= 300 && res.status < 400 && loc && /^https:\/\//.test(loc)) {
      try {
        const next = await this.fetchFn(loc, { method: 'HEAD', redirect: 'follow', headers: { 'User-Agent': 'rulz-llm-suite' } })
        size = num(next.headers.get('content-length'))
      } catch {
        // the script asks again
      }
    }
    if (!gated) return { size, sha256 }
    const redirect = res.status >= 300 && loc ? parseUrl(loc, url) : null
    return { size, sha256, signedUrl: redirect && redirect.protocol === 'https:' && !isHfHost(redirect) ? redirect.href : '' }
  }

  /** A HEAD request to huggingface.co, following its redirects only within that host. */
  private async hfHead(url: string, token: string | null): Promise<Response> {
    const headers: Record<string, string> = { 'User-Agent': 'rulz-llm-suite' }
    if (token) headers.Authorization = `Bearer ${token}`
    let at = url
    for (let hop = 0; ; hop++) {
      const res = await this.fetchFn(at, { method: 'HEAD', redirect: 'manual', headers })
      const loc = res.status >= 300 && res.status < 400 ? res.headers.get('location') : null
      const next = loc ? parseUrl(loc, at) : null
      // Only a redirect to another address of the Hub itself; the one to the file's storage is the answer.
      if (!next || !isHfHost(next) || res.headers.get('x-linked-size') || hop >= 3) return res
      at = next.href
    }
  }

  private async resolveHf(files: RemoteFile[]): Promise<void> {
    const token = (await this.deps.secrets.hfToken())?.value ?? null
    const todo = files.filter((f) => f.url && (f.size === null || f.sha256 === null))
    let next = 0
    const worker = async (): Promise<void> => {
      while (next < todo.length) {
        const f = todo[next++]
        const h = await this.headFile(f.url as string, token)
        f.size ??= h.size
        f.sha256 ??= h.sha256
      }
    }
    await Promise.all([worker(), worker(), worker(), worker()])
  }

  async missingBytes(rentalId: number, def: RemoteModelDef): Promise<number | null> {
    const files = this.filesOf(def)
    await this.resolveHf(files)
    const res = await this.deps.ssh.run(rentalId, modelStateScript(slugOf(def.id)), { timeoutMs: 30_000 })
    const have = new Map<string, number>()
    const part = new Map<string, number>()
    for (const v of markers(res.stdout, 'FILE')) {
      const [n, s] = v.split('|')
      have.set(n, Number(s))
    }
    for (const v of markers(res.stdout, 'PART')) {
      const [n, s] = v.split('|')
      part.set(n.replace(/\.part$/, ''), Number(s))
    }
    const complete = new Map(parseComplete(Buffer.from(marker(res.stdout, 'COMPLETE_B64') ?? '', 'base64').toString('utf8')).map((e) => [e.name, e]))
    let missing = 0
    for (const f of files) {
      if (f.size === null) return null
      const c = complete.get(f.name)
      if (recordedComplete(f, c, have.get(f.name), f.localPath ? { sha256: cachedHash(f.localPath, this.cache), source: localSource(f.localPath) } : undefined)) continue
      // A file recorded from another address is fetched again from the start: its part file is not reused either.
      const otherSource = !!(f.url && c && c.source !== '-' && c.source !== f.url)
      missing += otherSource ? f.size : Math.max(0, f.size - Math.min(f.size, part.get(f.name) ?? 0))
    }
    return missing
  }

  async transferModel(rentalId: number, def: RemoteModelDef, onProgress: (p: TransferProgress) => void, signal: AbortSignal): Promise<void> {
    const slug = slugOf(def.id)
    const files = this.filesOf(def)
    const end = this.deps.directory.use(rentalId, 'setup', 'window')
    const t = this.track(rentalId, signal)
    let downloading = false
    try {
      onProgress({ stage: 'Looking at the files', bytesDone: 0, bytesTotal: 0 })
      await this.resolveHf(files)
      const total = files.reduce((a, f) => a + (f.size ?? 0), 0)
      let base = 0
      let stage = 'Starting'
      const report = (current: number, s = stage): void => {
        stage = s
        onProgress({ stage, bytesDone: base + current, bytesTotal: total })
      }
      const shas = new Map<string, string>()
      const sources = new Map<string, string>()
      for (const f of files.filter((x) => x.localPath)) {
        // Before the file is read: one that is replaced meanwhile then counts as changed.
        const source = localSource(f.localPath as string)
        if (source) sources.set(f.name, source)
        const up = await uploadFile({
          ssh: this.deps.ssh,
          rentalId,
          slug,
          remoteName: f.name,
          localPath: f.localPath as string,
          cache: this.cache,
          signal: t.ctl.signal,
          onProgress: (p) => report(p.bytesDone, p.stage)
        })
        shas.set(f.name, up.sha256)
        base += up.size
        report(0)
      }
      const downloads = files.filter((f) => f.url)
      if (downloads.length) {
        downloading = true
        const sizes = new Map(downloads.map((f) => [f.name, f.size ?? 0]))
        const doneFiles = new Set<string>()
        // The bar never goes back inside the downloads, and the stage says how fast they go.
        let top = base
        let meter = new RateMeter()
        const send = (done: number): void => {
          top = Math.max(top, done)
          meter.add(top)
          const mbps = meter.mbps()
          onProgress({ stage: mbps === null ? stage : `${stage}, ${mbps} Mbps`, bytesDone: top, bytesTotal: total })
        }
        const startStage = (label: string): void => {
          stage = label
          // A new meter: what an earlier try left in a part file must not look like speed.
          meter = new RateMeter()
          onProgress({ stage, bytesDone: top, bytesTotal: total })
        }
        const fetchOnto = (list: RemoteFile[], signedUrl: string | null): Promise<RunResult> =>
          this.deps.ssh.run(
            rentalId,
            hfDownloadScript(
              slug,
              list.map((f) => ({ url: f.url as string, name: f.name, size: f.size, sha256: f.sha256 }))
            ),
            {
              stdin: hfConfigLine(signedUrl),
              timeoutMs: 0,
              signal: t.ctl.signal,
              onLine: (line) => {
                const m = /^@rz (STAGE|PROG|FILE_DONE)=(.*)$/.exec(line)
                if (!m) return
                if (m[1] === 'STAGE') startStage(m[2])
                else if (m[1] === 'PROG') {
                  const d = m[2].split('|')[1]
                  const finished = [...doneFiles].reduce((a, n) => a + (sizes.get(n) ?? 0), 0)
                  send(base + finished + (Number(d) || 0))
                } else doneFiles.add(m[2])
              }
            }
          )
        const token = (await this.deps.secrets.hfToken())?.value ?? null
        const open: RemoteFile[] = []
        const gated: RemoteFile[] = []
        for (const f of downloads) {
          const h = token ? await this.headFile(f.url as string, token) : null
          ;(h?.signedUrl === undefined ? open : gated).push(f)
        }
        if (open.length) {
          const res = await fetchOnto(open, null)
          if (res.code !== 0) throw this.failure(res, 'The download failed.')
        }
        for (const f of gated) {
          // A fresh address for each try: they expire while a long download runs; the next try resumes.
          for (let tries = 1; ; tries++) {
            t.ctl.signal.throwIfAborted()
            const { signedUrl } = await this.headFile(f.url as string, token)
            if (!signedUrl) throw new Error(`Hugging Face gives no download address for ${f.name} that works without your token, and the token is not sent to the instance. Download it to this PC and add it from here.`)
            const res = await fetchOnto([f], signedUrl)
            if (res.code === 0) break
            if (res.code !== 36 && res.code !== 35) throw this.failure(res, 'The download failed.')
            if (tries >= GATED_TRIES) throw this.failure(res, 'The download failed.')
          }
        }
        base += downloads.reduce((a, f) => a + (f.size ?? 0), 0)
      }
      report(0, 'Checking the files')
      const state = await this.deps.ssh.run(rentalId, modelStateScript(slug), { timeoutMs: 30_000, signal: t.ctl.signal })
      const sizeOf = new Map<string, number>()
      for (const v of markers(state.stdout, 'FILE')) {
        const [n, s] = v.split('|')
        sizeOf.set(n, Number(s))
      }
      const entries = files.map((f) => {
        const size = sizeOf.get(f.name)
        if (size === undefined) throw new Error(`${f.name} is missing on the instance after the transfer.`)
        if (f.size !== null && f.size !== size) throw new Error(`${f.name} has ${size} bytes on the instance, expected ${f.size}.`)
        return { name: f.name, size, sha256: shas.get(f.name) ?? f.sha256 ?? '-', source: f.url ?? sources.get(f.name) ?? '-' }
      })
      const mark = await this.deps.ssh.run(rentalId, markCompleteScript(slug, entries), { timeoutMs: 30_000, signal: t.ctl.signal })
      if (mark.code !== 0) throw this.failure(mark, 'The files could not be marked complete.')
      onProgress({ stage: 'Model ready', bytesDone: total, bytesTotal: total })
    } catch (err) {
      // A cancelled setup or a closed connection leaves what runs on the instance running: ask it to stop.
      if (downloading && t.ctl.signal.aborted) await this.stopDownloads(rentalId, slug)
      throw err
    } finally {
      t.done()
      end()
    }
  }

  /** Best effort, with its own short limit and not the cancelled signal: ends what a cancelled setup left running. */
  private async stopDownloads(rentalId: number, slug: string): Promise<void> {
    try {
      const res = await this.deps.ssh.run(rentalId, hfStopScript(slug), { timeoutMs: 25_000 })
      if (res.code !== 0) this.note(rentalId, `Could not stop the download on the instance: ${marker(res.stdout, 'ERR') ?? `exit ${res.code}`}`)
    } catch (err) {
      this.note(rentalId, `Could not stop the download on the instance: ${(err as Error).message}`)
    }
  }

  async removeModelFiles(rentalId: number, modelId: string): Promise<void> {
    const res = await this.deps.ssh.run(rentalId, removeModelScript(slugOf(modelId)), { timeoutMs: 120_000 })
    if (res.code !== 0) throw this.failure(res, 'The model files could not be removed.')
  }

  // ---------- Keepalive ----------

  async touch(rentalId: number): Promise<void> {
    await this.deps.ssh.run(rentalId, touchScript(), { timeoutMs: 20_000 })
  }

  setInUse(rentalId: number, inUse: boolean): void {
    const cur = this.keeps.get(rentalId)
    if (!inUse) {
      if (cur) this.stopKeep(rentalId, cur)
      return
    }
    if (cur?.active) return
    const k: KeepAlive = { active: true, proc: null, ticker: null, retry: null, failures: 0 }
    this.keeps.set(rentalId, k)
    this.connectKeep(rentalId, k)
  }

  private connectKeep(rentalId: number, k: KeepAlive): void {
    k.retry = null
    if (!k.active) return
    let proc: SshProcess
    try {
      proc = this.deps.ssh.open(rentalId, keepaliveLoopScript())
    } catch (err) {
      this.scheduleKeep(rentalId, k, (err as Error).message)
      return
    }
    k.proc = proc
    const startedAt = Date.now()
    // The remote loop touches keepalive for every line; ending stdin (EOF) ends the loop, so nothing is left behind.
    k.ticker = setInterval(() => proc.write('\n'), KEEPALIVE_MS)
    void proc.done.then((res) => {
      if (k.ticker) clearInterval(k.ticker)
      k.ticker = null
      if (k.proc === proc) k.proc = null
      if (!k.active) return
      if (Date.now() - startedAt > 30_000) k.failures = 0
      if (res.error) this.deps.ssh.routeFailed?.(rentalId)
      this.scheduleKeep(rentalId, k, res.error?.message ?? 'The connection ended.')
    })
  }

  private scheduleKeep(rentalId: number, k: KeepAlive, why: string): void {
    if (!k.active) return
    k.failures++
    const wait = Math.min(60_000, 2000 * 2 ** Math.min(k.failures - 1, 5))
    if (k.failures === 1 || k.failures % 5 === 0) this.note(rentalId, `Keepalive connection lost (${why}); trying again in ${Math.round(wait / 1000)} s.`)
    k.retry = setTimeout(() => this.connectKeep(rentalId, k), wait)
  }

  private stopKeep(rentalId: number, k: KeepAlive, killNow = false): void {
    k.active = false
    if (k.ticker) clearInterval(k.ticker)
    if (k.retry) clearTimeout(k.retry)
    k.ticker = null
    k.retry = null
    const proc = k.proc
    k.proc = null
    this.keeps.delete(rentalId)
    if (proc) {
      proc.end()
      // EOF ends the remote loop by itself; the kill is for a connection that is stuck (at once when disconnecting).
      if (killNow) proc.kill()
      else setTimeout(() => proc.kill(), 3000).unref?.()
    }
  }

  disconnect(rentalId: number): void {
    const k = this.keeps.get(rentalId)
    if (k) {
      this.stopKeep(rentalId, k, true)
    }
    for (const ctl of this.running.get(rentalId) ?? []) ctl.abort()
    this.running.delete(rentalId)
  }
}

