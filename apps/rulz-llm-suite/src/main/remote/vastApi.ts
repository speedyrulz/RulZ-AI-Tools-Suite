// vast.ai REST client (docs/vast-design.md, sections 2 and 3). Only the rent call is dangerous: it has no idempotency key,
// so it is sent at most once and its failures are split into "definitely refused" and "may have been accepted".
import type { OfferFilter, RemoteOffer, VastInstance } from '@shared/remote'
import type { SshRoute } from './contract'

const BASE = 'https://console.vast.ai/api/v0'
const USER_AGENT = 'rulz-llm-suite'
const BACKOFF_SECONDS = [1, 2, 4, 8]
const READ_TIMEOUT_MS = 20_000
const RENT_TIMEOUT_MS = 30_000
const PAGE_SIZE = 25
const MAX_PAGES = 200
/** A "24 GB" card reports 24564 or 24000 MiB; allow a little under the round number. */
const VRAM_TOLERANCE = 0.98

export interface FetchResponse {
  status: number
  text(): Promise<string>
}
export type FetchLike = (url: string, init: { method: string; headers: Record<string, string>; body?: string; signal?: AbortSignal }) => Promise<FetchResponse>

export interface VastApiDeps {
  key: () => Promise<string | null>
  fetch?: FetchLike
  sleep?: (ms: number) => Promise<void>
  now?: () => number
}

export type VastErrorKind = 'no-key' | 'auth' | 'rate' | 'client' | 'server' | 'network' | 'unreadable'

export class VastApiError extends Error {
  constructor(
    message: string,
    readonly kind: VastErrorKind,
    /** True when vast.ai certainly did not act on the request (so a rent was not made). */
    readonly definite: boolean,
    readonly status?: number
  ) {
    super(message)
    this.name = 'VastApiError'
  }
}

export interface VastInstanceDetail extends VastInstance {
  /** vast's intended_status (running, stopped, ...), '' = not reported. */
  intended: string
  routes: SshRoute[]
}

/** What the rental manager needs of the client; tests pass a fake. */
export interface VastApiClient {
  searchOffers(filter: OfferFilter): Promise<RemoteOffer[]>
  rent(req: { offerId: number; image: string; diskGb: number; label: string; onstart: string }): Promise<{ contractId: number }>
  showInstance(id: number): Promise<VastInstanceDetail | null>
  listInstances(): Promise<VastInstanceDetail[]>
  startInstance(id: number): Promise<void>
  stopInstance(id: number): Promise<void>
  destroyInstance(id: number): Promise<void>
  sshKeys(): Promise<string[]>
  addSshKey(publicKey: string): Promise<void>
  currentUser(): Promise<{ credit: number | null }>
}

// ---------- validation helpers ----------

export function num(v: unknown): number | null {
  if (typeof v === 'number') return Number.isFinite(v) ? v : null
  if (typeof v === 'string' && v.trim() !== '') {
    const n = Number(v)
    return Number.isFinite(n) ? n : null
  }
  return null
}
const clamp = (n: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, n))
const str = (v: unknown, max = 200): string => (typeof v === 'string' ? v.trim().slice(0, max) : '')
const posInt = (v: unknown): number | null => {
  const n = num(v)
  return n !== null && Number.isSafeInteger(n) && n > 0 ? n : null
}
const isObject = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v)

/** An offer filter from the window with every value forced into a sane range. */
export function cleanOfferFilter(raw: unknown): OfferFilter {
  const r = isObject(raw) ? raw : {}
  const n = (v: unknown, d: number, lo: number, hi: number): number => clamp(num(v) ?? d, lo, hi)
  const maxDph = r.maxDollarsPerHour === null ? null : num(r.maxDollarsPerHour)
  const gpuName = str(r.gpuName, 60)
  return {
    minVramGb: n(r.minVramGb, 24, 1, 2000),
    gpuName: /^[A-Za-z0-9 _.+-]*$/.test(gpuName) ? gpuName : '',
    maxGpus: Math.round(n(r.maxGpus, 1, 1, 16)),
    maxDollarsPerHour: maxDph === null ? null : clamp(maxDph, 0.01, 100),
    diskGb: Math.round(n(r.diskGb, 60, 10, 4000)),
    minReliability: n(r.minReliability, 0.98, 0, 1),
    minInetDownMbps: n(r.minInetDownMbps, 200, 0, 100_000)
  }
}

export function offerSearchBody(filter: OfferFilter): Record<string, unknown> {
  const f = cleanOfferFilter(filter)
  const body: Record<string, unknown> = {
    limit: 64,
    type: 'ondemand',
    order: [['dph_total', 'asc']],
    allocated_storage: f.diskGb,
    rentable: { eq: true },
    rented: { eq: false },
    verified: { eq: true },
    cpu_arch: { eq: 'amd64' },
    cuda_max_good: { gte: 12.8 },
    num_gpus: { lte: f.maxGpus },
    // gpu_ram is per GPU; the total is checked on the answer, so this only has to be a safe lower bound.
    gpu_ram: { gte: Math.floor((f.minVramGb * 1000 * VRAM_TOLERANCE) / f.maxGpus) },
    reliability: { gte: f.minReliability },
    inet_down: { gte: f.minInetDownMbps },
    disk_space: { gte: f.diskGb }
  }
  if (f.maxDollarsPerHour !== null) body.dph_total = { lte: f.maxDollarsPerHour }
  if (f.gpuName) body.gpu_name = { eq: f.gpuName }
  return body
}

export function parseOffer(raw: unknown): RemoteOffer | null {
  if (!isObject(raw)) return null
  const id = posInt(raw.id)
  const dph = num(raw.dph_total)
  const numGpus = posInt(raw.num_gpus)
  const gpuRam = num(raw.gpu_ram)
  if (id === null || dph === null || dph < 0 || numGpus === null || gpuRam === null || gpuRam <= 0) return null
  const dur = num(raw.duration)
  const perGb = (v: unknown): number | null => {
    const n = num(v)
    return n !== null && n >= 0 ? n : null
  }
  const rel = num(raw.reliability2 ?? raw.reliability)
  return {
    id,
    machineId: posInt(raw.machine_id) ?? 0,
    gpuName: str(raw.gpu_name, 80),
    numGpus,
    gpuRamMiB: gpuRam,
    cpuCores: Math.max(0, num(raw.cpu_cores_effective ?? raw.cpu_cores) ?? 0),
    cpuRamMiB: Math.max(0, num(raw.cpu_ram) ?? 0),
    dollarsPerHour: dph,
    reliability: rel === null ? 0 : clamp(rel, 0, 1),
    inetDownMbps: Math.max(0, num(raw.inet_down) ?? 0),
    inetUpMbps: Math.max(0, num(raw.inet_up) ?? 0),
    inetDownPerGb: perGb(raw.inet_down_cost),
    inetUpPerGb: perGb(raw.inet_up_cost),
    cudaMaxGood: Math.max(0, num(raw.cuda_max_good) ?? 0),
    geolocation: str(raw.geolocation, 80),
    maxDays: dur !== null && dur > 0 ? dur / 86400 : null
  }
}

const HOST_RE = /^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$/
const validPort = (n: number | null): n is number => n !== null && Number.isInteger(n) && n >= 1 && n <= 65535

/** Direct port first, then vast's proxy. The array-of-int form of `ports` means no direct route. */
export function parseRoutes(raw: Record<string, unknown>): SshRoute[] {
  const routes: SshRoute[] = []
  const ip = str(raw.public_ipaddr, 100)
  const ports = raw.ports
  if (ip && HOST_RE.test(ip) && isObject(ports)) {
    const list = ports['22/tcp']
    const first = Array.isArray(list) ? list.find((p) => isObject(p) && validPort(num(p.HostPort))) : null
    if (isObject(first)) routes.push({ route: 'direct', host: ip, port: num(first.HostPort) as number })
  }
  const host = str(raw.ssh_host, 253)
  const port = num(raw.ssh_port)
  if (host && HOST_RE.test(host) && validPort(port)) routes.push({ route: 'proxy', host, port })
  return routes
}

export function parseInstance(raw: unknown): VastInstanceDetail | null {
  if (!isObject(raw)) return null
  const id = posInt(raw.id)
  if (id === null) return null
  const disk = Math.max(0, num(raw.disk_space) ?? 0)
  const total = num(raw.storage_total_cost)
  const perGbMonth = num(raw.storage_cost)
  // storage_total_cost is already per hour for the whole disk; storage_cost is $/GB/month.
  const storage = total !== null && total >= 0 ? total : perGbMonth !== null && perGbMonth >= 0 && disk > 0 ? (perGbMonth * disk) / 730 : null
  const start = num(raw.start_date)
  const dph = num(raw.dph_total)
  return {
    id,
    label: str(raw.label, 120),
    status: str(raw.actual_status, 40).toLowerCase() || 'unknown',
    statusMessage: str(raw.status_msg, 400),
    gpuName: str(raw.gpu_name, 80),
    numGpus: Math.max(1, num(raw.num_gpus) ?? 1),
    gpuRamMiB: Math.max(0, num(raw.gpu_ram) ?? 0),
    dollarsPerHour: dph !== null && dph >= 0 ? dph : 0,
    storageDollarsPerHour: storage,
    diskGb: disk,
    startedAt: start !== null && start > 0 ? Math.round(start * 1000) : null,
    image: str(raw.image_uuid ?? raw.image, 200),
    intended: str(raw.intended_status, 40).toLowerCase(),
    routes: parseRoutes(raw)
  }
}

function messageOf(body: unknown): string {
  if (!isObject(body)) return ''
  for (const k of ['msg', 'error', 'detail', 'message']) {
    const v = body[k]
    if (typeof v === 'string' && v.trim()) return v.trim().slice(0, 200)
  }
  return ''
}

// ---------- client ----------

export class VastApi implements VastApiClient {
  private readonly fetchFn: FetchLike
  private readonly sleep: (ms: number) => Promise<void>

  constructor(private readonly deps: VastApiDeps) {
    this.fetchFn = deps.fetch ?? ((url, init) => fetch(url, init) as unknown as Promise<FetchResponse>)
    this.sleep = deps.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms)))
  }

  private async call(
    method: string,
    url: string,
    body: unknown,
    opts: { retry429: boolean; timeoutMs?: number }
  ): Promise<{ status: number; json: unknown; parsed: boolean }> {
    const key = await this.deps.key()
    if (!key) throw new VastApiError('Add your vast.ai API key first.', 'no-key', true)
    const headers: Record<string, string> = { Authorization: `Bearer ${key}`, Accept: 'application/json', 'User-Agent': USER_AGENT }
    if (body !== undefined) headers['Content-Type'] = 'application/json'
    for (let attempt = 0; ; attempt++) {
      let status: number
      let text: string
      try {
        const res = await this.withTimeout(opts.timeoutMs ?? READ_TIMEOUT_MS, (signal) =>
          this.fetchFn(url, { method, headers, body: body === undefined ? undefined : JSON.stringify(body), signal }).then(async (r) => ({ status: r.status, text: await r.text() }))
        )
        status = res.status
        text = res.text
      } catch (e) {
        const timedOut = (e as Error).message === 'timeout'
        throw new VastApiError(timedOut ? 'vast.ai did not answer in time.' : 'Could not reach vast.ai.', 'network', false)
      }
      let json: unknown
      let parsed = false
      try {
        json = JSON.parse(text)
        parsed = true
      } catch {
        json = undefined
      }
      if (status === 429) {
        const wait = BACKOFF_SECONDS[attempt]
        if (opts.retry429 && wait !== undefined) {
          await this.sleep(wait * 1000)
          continue
        }
        throw new VastApiError('vast.ai says there are too many requests right now. Try again in a minute.', 'rate', true, 429)
      }
      if (status === 401 || status === 403) {
        throw new VastApiError('vast.ai refused the API key. Check it on the vast.ai page.', 'auth', true, status)
      }
      if (status >= 500) throw new VastApiError(`vast.ai had an error (${status}).`, 'server', false, status)
      if (status >= 400) {
        const msg = messageOf(json)
        throw new VastApiError(msg ? `vast.ai said no: ${msg}` : `vast.ai answered ${status}.`, 'client', parsed && isObject(json), status)
      }
      return { status, json, parsed }
    }
  }

  private async withTimeout<T>(ms: number, run: (signal: AbortSignal) => Promise<T>): Promise<T> {
    const ac = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined
    const timeout = new Promise<never>((_, reject) => {
      timer = setTimeout(() => {
        ac.abort()
        reject(new Error('timeout'))
      }, ms)
    })
    try {
      return await Promise.race([run(ac.signal), timeout])
    } finally {
      clearTimeout(timer)
    }
  }

  private readJson(r: { json: unknown; parsed: boolean }): unknown {
    if (!r.parsed) throw new VastApiError('vast.ai sent an answer the app could not read.', 'unreadable', false)
    return r.json
  }

  async searchOffers(filter: OfferFilter): Promise<RemoteOffer[]> {
    const f = cleanOfferFilter(filter)
    const json = this.readJson(await this.call('POST', `${BASE}/bundles/`, offerSearchBody(f), { retry429: true }))
    const raw = isObject(json) ? json.offers : json
    const list = Array.isArray(raw) ? raw : isObject(raw) ? [raw] : []
    const need = f.minVramGb * 1000 * VRAM_TOLERANCE
    const out: RemoteOffer[] = []
    for (const r of list) {
      const o = parseOffer(r)
      if (o && o.gpuRamMiB * o.numGpus >= need && o.numGpus <= f.maxGpus) out.push(o)
    }
    return out
  }

  /** Sent at most once, whatever happens. A 429 here means refused: nothing was rented. */
  async rent(req: { offerId: number; image: string; diskGb: number; label: string; onstart: string }): Promise<{ contractId: number }> {
    const id = posInt(req.offerId)
    if (id === null) throw new VastApiError('That offer id is not valid.', 'client', true)
    const body = {
      image: req.image,
      disk: req.diskGb,
      label: req.label,
      runtype: 'ssh_direct',
      target_state: 'running',
      cancel_unavail: true,
      onstart: req.onstart,
      env: {}
    }
    const r = await this.call('PUT', `${BASE}/asks/${id}/`, body, { retry429: false, timeoutMs: RENT_TIMEOUT_MS })
    if (!r.parsed || !isObject(r.json)) throw new VastApiError('vast.ai answered the rent request with something unreadable.', 'unreadable', false)
    if (r.json.success === false) {
      throw new VastApiError(`vast.ai refused the rent: ${messageOf(r.json) || 'no reason given'}.`, 'client', true, r.status)
    }
    const contract = posInt(r.json.new_contract)
    if (r.json.success !== true || contract === null) {
      throw new VastApiError('vast.ai answered the rent request without a contract number.', 'unreadable', false)
    }
    return { contractId: contract }
  }

  async showInstance(id: number): Promise<VastInstanceDetail | null> {
    const n = posInt(id)
    if (n === null) throw new VastApiError('That instance id is not valid.', 'client', true)
    let r: { json: unknown; parsed: boolean }
    try {
      r = await this.call('GET', `${BASE}/instances/${n}/`, undefined, { retry429: true })
    } catch (e) {
      if (e instanceof VastApiError && e.status === 404) return null
      throw e
    }
    const json = this.readJson(r)
    const raw = isObject(json) ? json.instances : null
    const inst = parseInstance(Array.isArray(raw) ? raw[0] : raw)
    return inst && inst.id === n ? inst : null
  }

  /** Every instance of the account; throws unless every page was read. */
  async listInstances(): Promise<VastInstanceDetail[]> {
    const base = BASE.replace('/api/v0', '/api/v1')
    const all = new Map<number, VastInstanceDetail>()
    let token = ''
    for (let page = 0; page < MAX_PAGES; page++) {
      const url = `${base}/instances/?limit=${PAGE_SIZE}${token ? `&after_token=${encodeURIComponent(token)}` : ''}`
      const json = this.readJson(await this.call('GET', url, undefined, { retry429: true }))
      const raw = Array.isArray(json) ? json : isObject(json) ? json.instances : null
      if (!Array.isArray(raw) && !(isObject(json) && 'instances' in json)) {
        throw new VastApiError('vast.ai sent an instance list the app could not read.', 'unreadable', false)
      }
      for (const r of Array.isArray(raw) ? raw : []) {
        const inst = parseInstance(r)
        if (inst) all.set(inst.id, inst)
      }
      const next = isObject(json) ? str(json.next_token, 4000) : ''
      if (!next) return [...all.values()]
      if (next === token) break
      token = next
    }
    throw new VastApiError('vast.ai kept sending more pages of instances; the list may be incomplete.', 'unreadable', false)
  }

  private async put(id: number, state: 'running' | 'stopped'): Promise<void> {
    const n = posInt(id)
    if (n === null) throw new VastApiError('That instance id is not valid.', 'client', true)
    const r = await this.call('PUT', `${BASE}/instances/${n}/`, { state }, { retry429: true })
    if (isObject(r.json) && r.json.success === false) throw new VastApiError(`vast.ai refused: ${messageOf(r.json) || 'no reason given'}.`, 'client', true)
  }
  startInstance(id: number): Promise<void> {
    return this.put(id, 'running')
  }
  stopInstance(id: number): Promise<void> {
    return this.put(id, 'stopped')
  }
  async destroyInstance(id: number): Promise<void> {
    const n = posInt(id)
    if (n === null) throw new VastApiError('That instance id is not valid.', 'client', true)
    const r = await this.call('DELETE', `${BASE}/instances/${n}/`, undefined, { retry429: true })
    if (isObject(r.json) && r.json.success === false) throw new VastApiError(`vast.ai refused: ${messageOf(r.json) || 'no reason given'}.`, 'client', true)
  }

  async sshKeys(): Promise<string[]> {
    const json = this.readJson(await this.call('GET', `${BASE}/ssh/`, undefined, { retry429: true }))
    const raw = Array.isArray(json) ? json : isObject(json) ? (json.keys ?? json.ssh_keys) : null
    if (!Array.isArray(raw)) throw new VastApiError('vast.ai sent a key list the app could not read.', 'unreadable', false)
    const out: string[] = []
    for (const k of raw) {
      const v = typeof k === 'string' ? k : isObject(k) ? (k.key ?? k.public_key) : null
      if (typeof v === 'string' && v.trim()) out.push(v.trim())
    }
    return out
  }

  async addSshKey(publicKey: string): Promise<void> {
    await this.call('POST', `${BASE}/ssh/`, { ssh_key: publicKey.trim() }, { retry429: true })
  }

  async currentUser(): Promise<{ credit: number | null }> {
    const json = this.readJson(await this.call('GET', `${BASE}/users/current/`, undefined, { retry429: true }))
    if (!isObject(json)) return { credit: null }
    const c = num(json.credit ?? json.balance)
    return { credit: c }
  }
}
