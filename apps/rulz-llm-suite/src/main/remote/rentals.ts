// The rental list: renting, attaching, starting, stopping, destroying, setup jobs, cost and auto-stop
// (docs/vast-design.md, sections 2 and 5). Only rent, start, stop and destroy (and the app's own auto-stop and the quit
// dialog, which both end in a stop) ever call a billing API. Everything else reads.
import { randomBytes } from 'node:crypto'
import {
  ATTACHED_AUTO_STOP,
  DEFAULT_ENGINE_TAG,
  DEFAULT_REMOTE_SETTINGS,
  STOP_COUNTDOWN_SECONDS,
  isRemoteModelId,
  type AutoStop,
  type OfferFilter,
  type RemoteGpu,
  type RemoteModelDef,
  type RemoteOffer,
  type RemoteSettings,
  type RemoteState,
  type RentIntent,
  type Rental,
  type RentalModel,
  type RentalStatus,
  type RentRequest,
  type VastInstance
} from '@shared/remote'
import { readJson, writeJsonAtomic } from '../util/jsonFile'
import type { ProbeResult, RentalDirectory, RentalInfo, RentalOps, Secrets, SshRoute, TransferProgress } from './contract'
import { VastApiError, cleanOfferFilter, num, type VastApiClient, type VastInstanceDetail } from './vastApi'

const HOUR = 3_600_000
const MINUTE = 60_000
const LIST_IDLE_MS = 60_000
const LIST_BUSY_MS = 10_000
const PROBE_EVERY_MS = 60_000
const PROBE_RETRY_MS = 30_000
const CREDIT_EVERY_MS = 5 * MINUTE
const MISSES_FOR_GONE = 3
const NOT_STARTING_GRACE_MS = 60_000
const STOP_IGNORED_MS = 2 * MINUTE
const VETO_MS = 10 * MINUTE
const FLAP_MS = 10 * MINUTE
const BUSY_PERCENT = 10
const KEPT = 'it is to keep running.'
const PROGRESS_EVERY_MS = 250
const MAX_LOG_LINES = 500
const KEEP_FREE_BYTES = 5e9
const ENGINE_BYTES = 2.5e9
/** Only float noise: any real rise above the confirmed price refuses the rent. */
const PRICE_RISE_REFUSE = 1.000001
const PRICE_RISE_REPORT = 1.02
/** Seconds after a lost rent answer at which the account is searched for the label, then once a minute to 10 minutes. */
const CHECK_OFFSETS = [5, 15, 30, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600]

const TAG_RE = /^b\d{3,7}$/
const NAME = '[a-z0-9]+(?:[._-][a-z0-9]+)*'
const IMAGE_RE = new RegExp(`^${NAME}(?:/${NAME})*(?::@?[A-Za-z0-9_][A-Za-z0-9_.@-]{0,127})?(?:@sha256:[a-f0-9]{64})?$`)
export const isEngineTag = (v: unknown): v is string => typeof v === 'string' && TAG_RE.test(v)
export const isDockerImage = (v: unknown): v is string => typeof v === 'string' && v.length <= 200 && IMAGE_RE.test(v)

export interface SshKeyInfo {
  available: boolean
  publicKeyPath: string | null
  publicKey: string | null
  error?: string
}

export interface RentalManagerDeps {
  api: VastApiClient
  ops: RentalOps
  secrets: Secrets
  models: { list(): RemoteModelDef[] }
  /** remote.json */
  statePath: string
  localEngineBuild: () => Promise<number | null>
  sshKey: () => Promise<SshKeyInfo>
  now?: () => number
  onChange?: (state: RemoteState) => void
  /** Cancels use of a rental cleanly (unloads instances, stops setup) before the app stops or destroys it. */
  onStopping?: (rentalId: number) => Promise<void>
  /** A new `rulz-llm-<8 hex>` label. */
  newLabel?: () => string
  setTimer?: (fn: () => void, ms: number) => unknown
  clearTimer?: (handle: unknown) => void
  stopHookTimeoutMs?: number
}

// ---------- settings and validation shared with ipc.ts ----------

/** `base` with `patch` applied; throws a sentence when a value is out of range. */
export function cleanAutoStop(patch: unknown, base: AutoStop): AutoStop {
  const p = patch && typeof patch === 'object' ? (patch as Record<string, unknown>) : {}
  const out: AutoStop = { ...base }
  if (p.enabled !== undefined) {
    if (typeof p.enabled !== 'boolean') throw new Error('Auto-stop must be on or off.')
    out.enabled = p.enabled
  }
  if (p.apiCountsAsUse !== undefined) {
    if (typeof p.apiCountsAsUse !== 'boolean') throw new Error('The API and phone setting must be on or off.')
    out.apiCountsAsUse = p.apiCountsAsUse
  }
  if (p.idleMinutes !== undefined) {
    const n = p.idleMinutes
    if (typeof n !== 'number' || !Number.isInteger(n) || n < 5 || n > 1440) throw new Error('The idle time must be a whole number of minutes from 5 to 1440.')
    out.idleMinutes = n
  }
  if (p.capHours !== undefined) {
    const n = p.capHours
    if (n !== null && (typeof n !== 'number' || !Number.isInteger(n) || n < 1 || n > 720)) throw new Error('The hour limit must be a whole number from 1 to 720, or none.')
    out.capHours = n
  }
  return out
}

export function cleanSettings(raw: unknown): RemoteSettings {
  const r = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}
  const tag = typeof r.engineTag === 'string' ? r.engineTag.trim() : ''
  const image = typeof r.image === 'string' ? r.image.trim() : ''
  if (tag && !isEngineTag(tag)) throw new Error('The engine tag must look like b12407 (the letter b and 3 to 7 digits), or be empty.')
  if (!isDockerImage(image)) throw new Error('That does not look like a docker image name.')
  return { engineTag: tag, image }
}

// ---------- internal records ----------

interface Rec {
  inst: VastInstance
  origin: 'app' | 'attached'
  rentalStatus: RentalStatus
  routes: SshRoute[]
  ssh: Rental['ssh']
  engine: Rental['engine']
  models: RentalModel[]
  gpus: RemoteGpu[]
  diskFreeBytes: number | null
  foreignProcesses: boolean
  autoStop: AutoStop
  watchdog: Rental['watchdog']
  uses: Map<number, { what: string; by: 'window' | 'api' | 'phone' }>
  counted: boolean
  lastUsedAt: number | null
  capAt: number | null
  periodStart: number | null
  spent: number
  spentAt: number | null
  countdown: { at: number; reason: string; kind: 'idle' | 'cap' | 'batch' } | null
  finalizing: boolean
  suppressUntil: number
  pendingModelId: string
  missCount: number
  stopRequestedAt: number | null
  awaitingSince: number | null
  /** When it was last seen to leave a billing status; a short gap is not a new period. */
  leftBillingAt: number | null
  /** The price the rent was confirmed at, until the contract's own price has been compared with it. */
  agreedPrice: number | null
  /** Loaded from disk: the time the app was closed (ms), counted if the rental turns out to be still running. */
  gap: number
  destroySent: boolean
  /** Arming is done one at a time, each with the limits as they are then. */
  arming: Promise<void>
  armSeq: number
  setup: { abort: AbortController; modelId: string | null } | null
  bringingUp: boolean
  nextProbeAt: number
  lastProbeAt: number
  armedByProbe: boolean
  error: string
  notes: string[]
}

interface IntentRec extends RentIntent {
  autoStop: AutoStop
  modelId: string
  diskGb: number
}

interface DiskRental {
  inst: VastInstance
  origin: 'app' | 'attached'
  rentalStatus: RentalStatus
  autoStop: AutoStop
  capAt: number | null
  periodStart: number | null
  lastUsedAt: number | null
  spent: number
  spentAt: number | null
  pendingModelId: string
  models: RentalModel[]
  agreedPrice: number | null
  destroySent: boolean
}

interface DiskState {
  version: 1
  rentals: DiskRental[]
  intents: IntentRec[]
  settings: RemoteSettings
}

const isBilling = (s: RentalStatus): boolean => s === 'running' || s === 'starting' || s === 'stopping'
const asNum = (v: unknown, d = 0): number => num(v) ?? d
const strOf = (v: unknown, max: number): string => (typeof v === 'string' ? v.slice(0, max) : '')

function cleanInst(raw: unknown): VastInstance | null {
  if (!raw || typeof raw !== 'object') return null
  const r = raw as Record<string, unknown>
  const id = num(r.id)
  if (id === null || !Number.isSafeInteger(id) || id <= 0) return null
  const opt = (v: unknown): number | null => {
    const n = num(v)
    return n !== null && n >= 0 ? n : null
  }
  return {
    id,
    label: strOf(r.label, 120),
    status: strOf(r.status, 40) || 'unknown',
    statusMessage: strOf(r.statusMessage, 400),
    gpuName: strOf(r.gpuName, 80),
    numGpus: Math.max(1, asNum(r.numGpus, 1)),
    gpuRamMiB: Math.max(0, asNum(r.gpuRamMiB)),
    dollarsPerHour: Math.max(0, asNum(r.dollarsPerHour)),
    storageDollarsPerHour: opt(r.storageDollarsPerHour),
    diskGb: Math.max(0, asNum(r.diskGb)),
    startedAt: opt(r.startedAt),
    image: strOf(r.image, 200)
  }
}

const STATUSES: RentalStatus[] = ['starting', 'running', 'stopping', 'stopped', 'gone', 'unknown']
const SETUP_STATES = ['none', 'working', 'ready', 'failed']

function cleanDiskRental(raw: unknown): DiskRental | null {
  if (!raw || typeof raw !== 'object') return null
  const r = raw as Record<string, unknown>
  const inst = cleanInst(r.inst)
  if (!inst || (r.origin !== 'app' && r.origin !== 'attached')) return null
  let autoStop: AutoStop
  try {
    autoStop = cleanAutoStop(r.autoStop, r.origin === 'app' ? { enabled: true, idleMinutes: 30, capHours: 12, apiCountsAsUse: false } : ATTACHED_AUTO_STOP)
  } catch {
    // An unusable limit on disk falls back to the safe side: the app's defaults for its own rentals.
    autoStop = r.origin === 'app' ? { enabled: true, idleMinutes: 30, capHours: 12, apiCountsAsUse: false } : { ...ATTACHED_AUTO_STOP }
  }
  const time = (v: unknown): number | null => {
    const n = num(v)
    return n !== null && n > 0 ? n : null
  }
  const models: RentalModel[] = []
  for (const m of Array.isArray(r.models) ? r.models : []) {
    const x = m as Record<string, unknown> | null
    if (x && isRemoteModelId(x.modelId as string) && typeof x.modelId === 'string' && x.modelId.length <= 80 && SETUP_STATES.includes(x.state as string)) {
      // A job that was running when the app closed is not running now.
      const state = x.state === 'working' ? 'none' : (x.state as RentalModel['state'])
      models.push({ modelId: x.modelId, state, bytesDone: 0, bytesTotal: 0, stage: '' })
    }
  }
  const pending = typeof r.pendingModelId === 'string' && r.pendingModelId.length <= 80 && isRemoteModelId(r.pendingModelId) ? r.pendingModelId : ''
  return {
    inst,
    origin: r.origin,
    rentalStatus: STATUSES.includes(r.rentalStatus as RentalStatus) ? (r.rentalStatus as RentalStatus) : 'unknown',
    autoStop,
    capAt: time(r.capAt),
    periodStart: time(r.periodStart),
    lastUsedAt: time(r.lastUsedAt),
    spent: Math.max(0, asNum(r.spent)),
    spentAt: time(r.spentAt),
    agreedPrice: time(r.agreedPrice),
    destroySent: r.destroySent === true,
    pendingModelId: pending,
    models
  }
}

function cleanIntent(raw: unknown): IntentRec | null {
  if (!raw || typeof raw !== 'object') return null
  const r = raw as Record<string, unknown>
  const offerId = num(r.offerId)
  const dph = num(r.dollarsPerHour)
  if (typeof r.label !== 'string' || !/^rulz-llm-[0-9a-f]{8}$/.test(r.label) || offerId === null || !Number.isSafeInteger(offerId) || dph === null || dph < 0) return null
  let autoStop: AutoStop
  try {
    autoStop = cleanAutoStop(r.autoStop, { enabled: true, idleMinutes: 30, capHours: 12, apiCountsAsUse: false })
  } catch {
    autoStop = { enabled: true, idleMinutes: 30, capHours: 12, apiCountsAsUse: false }
  }
  const state = r.state === 'unresolved' ? 'unresolved' : 'checking'
  return {
    label: r.label,
    offerId,
    dollarsPerHour: dph,
    createdAt: asNum(r.createdAt),
    state,
    message: strOf(r.message, 400),
    autoStop,
    modelId: typeof r.modelId === 'string' && isRemoteModelId(r.modelId) && r.modelId.length <= 80 ? r.modelId : '',
    diskGb: Math.max(10, asNum(r.diskGb, 60))
  }
}

const sentence = (e: unknown): string => {
  const m = e instanceof Error ? e.message : String(e)
  return m.trim() || 'Something went wrong.'
}

const sameKey = (a: string, b: string): boolean => {
  const x = a.trim().split(/\s+/).slice(0, 2).join(' ')
  const y = b.trim().split(/\s+/).slice(0, 2).join(' ')
  return !!x && x === y
}

export class RentalManager implements RentalDirectory {
  private readonly recs = new Map<number, Rec>()
  private intents: IntentRec[] = []
  private intentRt = new Map<string, { since: number; done: number }>()
  private settings: RemoteSettings = { ...DEFAULT_REMOTE_SETTINGS }
  private vastSource = ''
  private hfSource = ''
  private credit: number | null = null
  private creditAt = 0
  private sshInfo: RemoteState['ssh'] = { available: false, publicKeyPath: null, onAccount: null }
  private localBuild: number | null = null
  private listError: string | null = null
  private lastListAt = 0
  private listing: Promise<void> | null = null
  private ticking: Promise<void> | null = null
  private renting = false
  private rev = 0
  private useSeq = 0
  private timer: unknown = null
  private disposed = false
  private lastProgressEmit = 0
  private readonly jobs = new Set<Promise<void>>()

  constructor(private readonly deps: RentalManagerDeps) {}

  private now(): number {
    return (this.deps.now ?? Date.now)()
  }

  // ---------- loading, saving, state ----------

  /** Reads remote.json. A rent that was being sent when the app died is settled like a lost answer. */
  async load(): Promise<void> {
    const raw = await readJson<unknown>(this.deps.statePath, null)
    const d = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}
    try {
      this.settings = cleanSettings(d.settings)
    } catch {
      this.settings = { ...DEFAULT_REMOTE_SETTINGS }
    }
    this.recs.clear()
    for (const x of Array.isArray(d.rentals) ? d.rentals : []) {
      const r = cleanDiskRental(x)
      if (r && !this.recs.has(r.inst.id)) this.recs.set(r.inst.id, this.fromDisk(r))
    }
    this.intents = []
    this.intentRt.clear()
    const now = this.now()
    for (const x of Array.isArray(d.intents) ? d.intents : []) {
      const i = cleanIntent(x)
      if (!i || this.intents.some((y) => y.label === i.label)) continue
      if (i.state !== 'unresolved') {
        i.state = 'checking'
        i.message = 'The app closed while the rent request was open. It may have been rented: checking your vast.ai instances for it.'
        this.intentRt.set(i.label, { since: now, done: 0 })
      }
      this.intents.push(i)
    }
    await this.refreshLocalFacts()
    if (this.intents.length) await this.saveNow().catch(() => undefined)
  }

  private fromDisk(r: DiskRental): Rec {
    const rec = this.blankRec(r.inst, r.origin, r.autoStop)
    rec.rentalStatus = r.rentalStatus
    rec.capAt = r.capAt
    rec.periodStart = r.periodStart
    rec.lastUsedAt = r.lastUsedAt
    rec.spent = r.spent
    rec.spentAt = isBilling(r.rentalStatus) ? this.now() : null
    rec.gap = isBilling(r.rentalStatus) && r.spentAt !== null ? Math.max(0, this.now() - r.spentAt) : 0
    rec.agreedPrice = r.agreedPrice
    rec.destroySent = r.destroySent
    rec.pendingModelId = r.pendingModelId
    rec.models = r.models
    return rec
  }

  private blankRec(inst: VastInstance, origin: 'app' | 'attached', autoStop: AutoStop): Rec {
    return {
      inst,
      origin,
      rentalStatus: 'unknown',
      routes: [],
      ssh: { state: 'unknown' },
      engine: { state: 'none', tag: '', stage: '' },
      models: [],
      gpus: [],
      diskFreeBytes: null,
      foreignProcesses: false,
      autoStop,
      watchdog: { state: 'unknown' },
      uses: new Map(),
      counted: false,
      lastUsedAt: null,
      capAt: null,
      periodStart: null,
      spent: 0,
      spentAt: null,
      countdown: null,
      finalizing: false,
      suppressUntil: 0,
      pendingModelId: '',
      missCount: 0,
      stopRequestedAt: null,
      awaitingSince: null,
      leftBillingAt: null,
      agreedPrice: null,
      gap: 0,
      destroySent: false,
      arming: Promise.resolve(),
      armSeq: 0,
      setup: null,
      bringingUp: false,
      nextProbeAt: 0,
      lastProbeAt: 0,
      armedByProbe: false,
      error: '',
      notes: []
    }
  }

  private toDisk(): DiskState {
    return {
      version: 1,
      settings: this.settings,
      intents: this.intents.map((i) => ({ ...i })),
      rentals: [...this.recs.values()].map((r) => ({
        inst: r.inst,
        origin: r.origin,
        rentalStatus: r.rentalStatus,
        autoStop: r.autoStop,
        capAt: r.capAt,
        periodStart: r.periodStart,
        lastUsedAt: r.lastUsedAt,
        spent: r.spent,
        spentAt: r.spentAt,
        pendingModelId: r.pendingModelId,
        models: r.models.filter((m) => m.state !== 'none'),
        agreedPrice: r.agreedPrice,
        destroySent: r.destroySent
      }))
    }
  }

  /** Awaited before anything that bills (the rent intent); the write is fsynced by writeJsonAtomic. */
  private saveNow(): Promise<void> {
    return writeJsonAtomic(this.deps.statePath, this.toDisk())
  }

  private save(): void {
    // Tracked as a job so idle() waits for the write.
    this.spawn(() => this.saveNow())
  }

  tag(): string {
    if (this.settings.engineTag && isEngineTag(this.settings.engineTag)) return this.settings.engineTag
    const b = this.localBuild
    if (b !== null && Number.isInteger(b) && isEngineTag(`b${b}`)) return `b${b}`
    return DEFAULT_ENGINE_TAG
  }

  private spawn(fn: () => Promise<void>): void {
    const p = fn()
      .catch((e) => console.error('Remote job failed:', this.msg(e)))
      .finally(() => this.jobs.delete(p))
    this.jobs.add(p)
  }

  /** Resolves when every background job (bring-up, setup, probe) has ended; for tests and for quitting. */
  async idle(): Promise<void> {
    while (this.jobs.size) await Promise.all([...this.jobs])
  }

  private buildRental(id: number, r: Rec): Rental {
    const running = r.rentalStatus === 'running'
    const counted = this.countedUses(r)
    const idleOn = running && r.autoStop.enabled && counted === 0
    const idleStopAt = idleOn ? (r.lastUsedAt ?? r.periodStart ?? this.now()) + r.autoStop.idleMinutes * MINUTE : null
    return {
      ...r.inst,
      id,
      origin: r.origin,
      rentalStatus: r.rentalStatus,
      ssh: { ...r.ssh },
      engine: { ...r.engine },
      models: r.models.map((m) => ({ ...m })),
      gpus: r.gpus.map((g) => ({ ...g })),
      diskFreeBytes: r.diskFreeBytes,
      foreignProcesses: r.foreignProcesses,
      autoStop: { ...r.autoStop },
      watchdog: { ...r.watchdog },
      inUse: [...r.uses.values()].map((u) => u.what),
      lastUsedAt: r.lastUsedAt,
      idleStopAt,
      capAt: (running || r.rentalStatus === 'starting') && r.autoStop.capHours !== null ? r.capAt : null,
      countdown: r.countdown ? { at: r.countdown.at, reason: r.countdown.reason } : null,
      spent: r.spent,
      ...(r.error ? { error: this.scrub(r.error) } : {})
    }
  }

  state(): RemoteState {
    const errs = [...this.recs.values()].filter((r) => r.error).map((r) => `${r.inst.label || `Instance ${r.inst.id}`}: ${r.error}`)
    return {
      rev: ++this.rev,
      rentals: [...this.recs.entries()].map(([id, r]) => this.buildRental(id, r)),
      intents: this.intents.map((i) => ({ label: i.label, offerId: i.offerId, dollarsPerHour: i.dollarsPerHour, createdAt: i.createdAt, state: i.state, message: i.message })),
      models: this.deps.models.list(),
      settings: { ...this.settings },
      vastKeySource: this.vastSource,
      hfTokenSource: this.hfSource,
      credit: this.credit,
      ssh: { ...this.sshInfo },
      resolvedEngineTag: this.tag(),
      error: this.listError ?? (errs.length ? this.scrub(errs.join(' ')) : null)
    }
  }

  private emit(): void {
    if (this.disposed || !this.deps.onChange) return
    try {
      this.deps.onChange(this.state())
    } catch (e) {
      console.error('onChange failed:', e)
    }
  }

  private scrub(text: string): string {
    return this.deps.secrets.scrub(text)
  }
  private msg(e: unknown): string {
    return this.scrub(sentence(e))
  }

  private async refreshLocalFacts(): Promise<void> {
    try {
      this.localBuild = await this.deps.localEngineBuild()
    } catch {
      this.localBuild = null
    }
    this.vastSource = (await this.deps.secrets.vastKey().catch(() => null))?.source ?? ''
    this.hfSource = (await this.deps.secrets.hfToken().catch(() => null))?.source ?? ''
    try {
      const k = await this.deps.sshKey()
      this.sshInfo = { available: k.available, publicKeyPath: k.publicKeyPath, onAccount: this.sshInfo.onAccount, ...(k.error ? { error: k.error } : {}) }
    } catch (e) {
      this.sshInfo = { available: false, publicKeyPath: null, onAccount: null, error: this.msg(e) }
    }
  }

  // ---------- notes and logs ----------

  note(rentalId: number, line: string): void {
    const rec = this.recs.get(rentalId)
    if (!rec) return
    const t = new Date(this.now()).toISOString().slice(11, 19)
    rec.notes.push(`${t} ${this.scrub(line)}`)
    if (rec.notes.length > MAX_LOG_LINES) rec.notes.splice(0, rec.notes.length - MAX_LOG_LINES)
  }

  log(rentalId: number): string[] {
    const rec = this.recs.get(rentalId)
    if (!rec) return []
    const lines = [...this.deps.ops.log(rentalId), ...rec.notes].map((l) => this.scrub(l))
    return lines.slice(-MAX_LOG_LINES)
  }

  // ---------- RentalDirectory ----------

  info(rentalId: number): RentalInfo | null {
    const r = this.recs.get(rentalId)
    if (!r || r.rentalStatus === 'gone') return null
    return {
      rentalId,
      label: r.inst.label,
      origin: r.origin,
      running: r.inst.status === 'running' && r.rentalStatus === 'running',
      routes: r.routes.map((x) => ({ ...x })),
      engineTag: this.tag(),
      dollarsPerHour: r.inst.dollarsPerHour
    }
  }

  private countedBy(r: Rec, by: 'window' | 'api' | 'phone'): boolean {
    return by === 'window' || r.autoStop.apiCountsAsUse
  }
  private countedUses(r: Rec): number {
    let n = 0
    for (const u of r.uses.values()) if (this.countedBy(r, u.by)) n++
    return n
  }

  private syncInUse(rentalId: number, r: Rec): void {
    const c = this.countedUses(r) > 0
    if (c === r.counted) return
    r.counted = c
    try {
      this.deps.ops.setInUse(rentalId, c)
    } catch (e) {
      console.error('setInUse failed:', e)
    }
  }

  use(rentalId: number, what: string, by: 'window' | 'api' | 'phone'): () => void {
    const r = this.recs.get(rentalId)
    if (!r) return () => undefined
    const token = ++this.useSeq
    r.uses.set(token, { what, by })
    if (this.countedBy(r, by)) {
      r.lastUsedAt = this.now()
      // Use ends a countdown that was only about being idle; a hard cap goes on.
      if (r.countdown && r.countdown.kind !== 'cap') {
        this.note(rentalId, 'It is in use again, so the stop was cancelled.')
        r.countdown = null
      }
      this.syncInUse(rentalId, r)
      void this.deps.ops.touch(rentalId).catch(() => undefined)
    }
    this.emit()
    let done = false
    return () => {
      if (done) return
      done = true
      const cur = this.recs.get(rentalId)
      if (!cur || !cur.uses.delete(token)) return
      if (this.countedBy(cur, by)) cur.lastUsedAt = this.now()
      this.syncInUse(rentalId, cur)
      this.emit()
    }
  }

  // ---------- instance data ----------

  private mapStatus(r: Rec, d: VastInstance): RentalStatus {
    const s = d.status
    const stopping = r.stopRequestedAt !== null
    if (s === 'running') return stopping ? 'stopping' : 'running'
    if (s === 'stopping') return 'stopping'
    if (s === 'exited' || s === 'stopped') return r.awaitingSince !== null ? 'starting' : 'stopped'
    if (['loading', 'created', 'creating', 'starting', 'scheduling', 'pending', 'connecting', 'rebooting'].includes(s)) return stopping ? 'stopping' : 'starting'
    return r.awaitingSince !== null ? 'starting' : 'unknown'
  }

  private accrue(r: Rec): void {
    const now = this.now()
    if (r.spentAt !== null && isBilling(r.rentalStatus)) r.spent += (r.inst.dollarsPerHour * Math.max(0, now - r.spentAt)) / HOUR
    r.spentAt = isBilling(r.rentalStatus) ? now : null
  }

  private beginPeriod(r: Rec, at: number): void {
    const now = this.now()
    r.periodStart = at
    r.spent = (r.inst.dollarsPerHour * Math.max(0, now - at)) / HOUR
    r.spentAt = now
    r.lastUsedAt = null
    r.capAt = r.autoStop.capHours !== null ? at + r.autoStop.capHours * HOUR : null
  }

  private applyInstance(rec: Rec, d: VastInstanceDetail | VastInstance): void {
    const prev = rec.rentalStatus
    this.accrue(rec)
    // An answer without a price says nothing about it; a rental is never free.
    const price = d.dollarsPerHour > 0 ? d.dollarsPerHour : rec.inst.dollarsPerHour
    const { id, label, status, statusMessage, gpuName, numGpus, gpuRamMiB, storageDollarsPerHour, diskGb, startedAt, image } = d
    // The label the app made stays as recorded; destroy compares it with what vast.ai shows now.
    rec.inst = { id, label: rec.origin === 'app' && rec.inst.label ? rec.inst.label : label, status, statusMessage, gpuName, numGpus, gpuRamMiB, dollarsPerHour: price, storageDollarsPerHour, diskGb, startedAt, image }
    if ('routes' in d) rec.routes = d.routes
    const now = this.now()
    if (rec.stopRequestedAt !== null) {
      if (status === 'exited' || status === 'stopped') rec.stopRequestedAt = null
      else if (status === 'running' && now - rec.stopRequestedAt > STOP_IGNORED_MS) {
        rec.stopRequestedAt = null
        rec.error = 'It is still running although a stop was sent.'
      }
    }
    let next = this.mapStatus(rec, d)
    if (rec.awaitingSince !== null) {
      if (status === 'running') rec.awaitingSince = null
      else if (['exited', 'stopped', 'offline', 'unknown'].includes(status) && now - rec.awaitingSince >= NOT_STARTING_GRACE_MS) {
        rec.awaitingSince = null
        rec.error = `vast.ai reports it as ${status} and it does not look like it will start. ${rec.origin === 'app' ? 'Destroy it to stop paying, or start it again.' : 'Start it again on vast.ai.'}`
        this.note(id, `Not starting: vast.ai reports ${status}.`)
        next = this.mapStatus(rec, d)
      }
    }
    rec.rentalStatus = next
    if (next === 'running' && prev !== 'running') {
      rec.ssh = { state: 'unknown' }
      rec.nextProbeAt = 0
      rec.watchdog = { state: 'unknown' }
      if (rec.error.startsWith('vast.ai reports') || rec.error.startsWith('It is still running')) rec.error = ''
    }
    if (next !== 'running' && prev === 'running') this.leftRunning(rec)
    if (isBilling(prev) && !isBilling(next)) rec.leftBillingAt = now
    if (isBilling(next) && !isBilling(prev) && rec.periodStart !== null && prev !== 'unknown') {
      // Started again elsewhere: a new period from now (vast's start_date is the contract's). A short gap is a status
      // that flapped: the limits go on as they were, so it neither gets a fresh cap nor loses hours that were added.
      const flap = rec.leftBillingAt !== null && now - rec.leftBillingAt < FLAP_MS
      if (!flap) this.beginPeriod(rec, now)
    }
    if (rec.gap > 0 && next !== 'unknown') {
      // It can only have cost money all along if it is still the same running period.
      if (isBilling(next) && isBilling(prev)) rec.spent += (price * rec.gap) / HOUR
      rec.gap = 0
    }
    if (rec.agreedPrice !== null && d.dollarsPerHour > 0) {
      if (d.dollarsPerHour > rec.agreedPrice * PRICE_RISE_REPORT) {
        rec.error = `The contract costs $${d.dollarsPerHour.toFixed(3)} per hour, more than the $${rec.agreedPrice.toFixed(3)} shown. Check it on vast.ai and destroy it if that is not acceptable.`
        this.note(id, rec.error)
      }
      rec.agreedPrice = null
    }
    if (!isBilling(next)) {
      rec.spentAt = null
      rec.countdown = null
    } else if (rec.spentAt === null) rec.spentAt = now
    if (rec.periodStart === null && isBilling(next)) this.beginPeriod(rec, startedAt ?? now)
  }

  private leftRunning(rec: Rec): void {
    rec.ssh = { state: 'unknown' }
    rec.gpus = []
    rec.foreignProcesses = false
    rec.watchdog = { state: 'unknown' }
    rec.countdown = null
    rec.counted = false
    rec.setup?.abort.abort()
    this.deps.ops.disconnect(rec.inst.id)
  }

  private markGone(rec: Rec): void {
    rec.rentalStatus = 'gone'
    rec.spentAt = null
    rec.countdown = null
    rec.error = 'vast.ai no longer lists this instance. It was probably destroyed. Remove it from the list.'
    rec.setup?.abort.abort()
    this.deps.ops.disconnect(rec.inst.id)
    this.note(rec.inst.id, 'The instance is not on the account any more.')
  }

  private applyProbe(rec: Rec, p: ProbeResult): void {
    rec.gpus = p.gpus.map((g) => ({ ...g }))
    rec.diskFreeBytes = p.diskFreeBytes
    rec.foreignProcesses = p.foreignProcesses
    rec.armedByProbe = p.watchdog.armed
    rec.ssh = { state: 'ok' }
    rec.lastProbeAt = this.now()
    if (!rec.setup) {
      rec.engine = { ...rec.engine, tag: this.tag(), state: p.engines.includes(this.tag()) ? 'ready' : 'none', stage: '' }
      const have = new Set(p.models)
      rec.models = rec.models.filter((m) => m.state === 'working' || have.has(m.modelId))
      for (const id of p.models) {
        const m = rec.models.find((x) => x.modelId === id)
        if (m) m.state = 'ready'
        else rec.models.push({ modelId: id, state: 'ready', bytesDone: 0, bytesTotal: 0, stage: '' })
      }
    }
  }

  // ---------- listing ----------

  private async refreshList(): Promise<void> {
    if (this.listing) return this.listing
    const p = this.doList().finally(() => {
      this.listing = null
    })
    this.listing = p
    return p
  }

  private async doList(): Promise<void> {
    let list: VastInstanceDetail[]
    try {
      list = await this.deps.api.listInstances()
    } catch (e) {
      // A failed list says nothing about the instances: never change a status on it.
      this.listError = this.msg(e)
      return
    }
    this.listError = null
    this.lastListAt = this.now()
    const by = new Map(list.map((i) => [i.id, i]))
    for (const rec of this.recs.values()) {
      const inst = by.get(rec.inst.id)
      if (inst) {
        rec.missCount = 0
        this.applyInstance(rec, inst)
        continue
      }
      // A fresh contract can be missing from the list for a moment: ask for it by id before counting a miss.
      let seen: VastInstanceDetail | null
      try {
        seen = await this.deps.api.showInstance(rec.inst.id)
      } catch {
        continue
      }
      if (seen) {
        rec.missCount = 0
        this.applyInstance(rec, seen)
      } else if (rec.destroySent) {
        this.recs.delete(rec.inst.id)
      } else if (++rec.missCount >= MISSES_FOR_GONE && rec.rentalStatus !== 'gone') this.markGone(rec)
    }
    if (this.now() - this.creditAt >= CREDIT_EVERY_MS) await this.readCredit()
    this.emit()
    this.save()
  }

  private async readCredit(): Promise<void> {
    this.creditAt = this.now()
    try {
      this.credit = (await this.deps.api.currentUser()).credit
    } catch {
      // The balance is a nicety; a failure here is not an account problem.
    }
  }

  private listDue(): boolean {
    if (!this.recs.size) return false
    const busy = [...this.recs.values()].some((r) => r.rentalStatus === 'starting' || r.rentalStatus === 'stopping')
    return this.now() - this.lastListAt >= (busy ? LIST_BUSY_MS : LIST_IDLE_MS)
  }

  async refresh(): Promise<RemoteState> {
    await this.refreshLocalFacts()
    try {
      const k = await this.deps.sshKey()
      if (k.available && k.publicKey) this.sshInfo.onAccount = (await this.deps.api.sshKeys()).some((a) => sameKey(a, k.publicKey as string))
    } catch {
      this.sshInfo.onAccount = null
    }
    await this.refreshList()
    await this.readCredit()
    this.emit()
    return this.state()
  }

  async instances(): Promise<VastInstance[]> {
    const list = await this.deps.api.listInstances()
    return list.map(({ intended: _i, routes: _r, ...inst }) => inst)
  }

  async searchOffers(filter: OfferFilter): Promise<RemoteOffer[]> {
    return this.deps.api.searchOffers(cleanOfferFilter(filter))
  }

  // ---------- keys and settings ----------

  async setVastKey(key: string): Promise<RemoteState> {
    await this.deps.secrets.setVastKey(key)
    await this.refreshLocalFacts()
    this.listError = null
    this.emit()
    return this.state()
  }

  async setHfToken(token: string): Promise<RemoteState> {
    await this.deps.secrets.setHfToken(token)
    await this.refreshLocalFacts()
    this.emit()
    return this.state()
  }

  async saveSettings(raw: RemoteSettings): Promise<RemoteState> {
    this.settings = cleanSettings(raw)
    await this.saveNow()
    this.emit()
    return this.state()
  }

  async addSshKey(): Promise<RemoteState> {
    const k = await this.deps.sshKey()
    if (!k.available || !k.publicKey) throw new Error(k.error || 'There is no ssh key on this PC to add. Create one with ssh-keygen first.')
    await this.deps.api.addSshKey(k.publicKey)
    this.sshInfo = { available: true, publicKeyPath: k.publicKeyPath, onAccount: true }
    this.emit()
    return this.state()
  }

  // ---------- rent ----------

  private newLabel(): string {
    return this.deps.newLabel ? this.deps.newLabel() : `rulz-llm-${randomBytes(4).toString('hex')}`
  }

  async rent(req: RentRequest): Promise<RemoteState> {
    const offerId = req?.offerId
    const shown = req?.shownDollarsPerHour
    if (!Number.isSafeInteger(offerId) || offerId <= 0) throw new Error('That offer is not valid.')
    if (typeof shown !== 'number' || !Number.isFinite(shown) || shown <= 0) throw new Error('The price shown is not valid.')
    const filter = cleanOfferFilter(req.filter)
    const autoStop = cleanAutoStop(req.autoStop, { enabled: true, idleMinutes: 30, capHours: 12, apiCountsAsUse: false })
    const modelId = typeof req.modelId === 'string' ? req.modelId : ''
    if (modelId && !this.deps.models.list().some((m) => m.id === modelId)) throw new Error('That remote model is not saved any more.')
    // No second rent while one is being sent or settled: the app cannot tell the two apart on vast.ai's side.
    if (this.renting || this.intents.length) {
      throw new Error('Another rent is still open. Wait until it is settled (or dismiss it once you have checked vast.ai) before renting again.')
    }
    this.renting = true
    try {
      return await this.rentInner({ offerId, shown, filter, autoStop, modelId })
    } finally {
      this.renting = false
    }
  }

  private async rentInner(a: { offerId: number; shown: number; filter: OfferFilter; autoStop: AutoStop; modelId: string }): Promise<RemoteState> {
    const api = this.deps.api
    if (!(await this.deps.secrets.vastKey())) throw new Error('Add your vast.ai API key first.')
    const ssh = await this.deps.sshKey()
    if (!ssh.available || !ssh.publicKey) throw new Error(ssh.error || 'There is no ssh key on this PC. Create one with ssh-keygen, then use Add ssh key.')
    let keys: string[]
    try {
      keys = await api.sshKeys()
    } catch (e) {
      throw new Error(`Could not check the ssh keys on your vast.ai account, so nothing was rented. ${this.msg(e)}`)
    }
    if (!keys.some((k) => sameKey(k, ssh.publicKey as string))) {
      throw new Error('Your ssh key is not on your vast.ai account yet, so the instance could not be reached. Use Add ssh key first. Nothing was rented.')
    }
    let offer: RemoteOffer | null
    try {
      offer = await api.findOffer(a.offerId, a.filter)
    } catch (e) {
      throw new Error(`Could not look the offer up again, so nothing was rented. ${this.msg(e)}`)
    }
    if (!offer) throw new Error('That offer is not available any more, or no longer fits the filter. Nothing was rented. Search again.')
    if (offer.dollarsPerHour > a.shown * PRICE_RISE_REFUSE) {
      throw new Error(`The price rose from $${a.shown.toFixed(3)} to $${offer.dollarsPerHour.toFixed(3)} per hour, so nothing was rented. Search again to see the current price.`)
    }
    const now = this.now()
    const intent: IntentRec = {
      label: this.newLabel(),
      offerId: offer.id,
      dollarsPerHour: offer.dollarsPerHour,
      createdAt: now,
      state: 'sending',
      message: 'The rent request is being sent.',
      autoStop: a.autoStop,
      modelId: a.modelId,
      diskGb: a.filter.diskGb
    }
    this.intents.push(intent)
    try {
      // On disk before the request leaves: a crash mid-request must be able to find the rental by its label.
      await this.saveNow()
    } catch (e) {
      this.intents = this.intents.filter((i) => i !== intent)
      throw new Error(`Could not save the rent record, so nothing was rented. ${this.msg(e)}`)
    }
    this.emit()
    let contractId: number
    try {
      contractId = (await api.rent({ offerId: offer.id, image: this.settings.image, diskGb: a.filter.diskGb, label: intent.label, onstart: this.deps.ops.onstartScript({ autoStop: a.autoStop }) })).contractId
    } catch (e) {
      if (e instanceof VastApiError && e.definite) {
        this.intents = this.intents.filter((i) => i !== intent)
        await this.saveNow().catch(() => undefined)
        this.emit()
        throw new Error(`${this.msg(e)} Nothing was rented.`)
      }
      // Never send it again: look for the label instead.
      intent.state = 'checking'
      intent.message = 'The rent request got no clear answer. It may have been rented: the app is checking your vast.ai instances for it. Do not rent again yet.'
      this.intentRt.set(intent.label, { since: this.now(), done: 0 })
      await this.saveNow().catch(() => undefined)
      this.emit()
      return this.state()
    }
    await this.adoptContract(intent, contractId, null)
    return this.state()
  }

  /** The rental exists: persist it, drop the intent, read back the real price. */
  private async adoptContract(intent: IntentRec, id: number, found: VastInstanceDetail | null): Promise<void> {
    let inst: VastInstanceDetail | null = found
    if (!inst) inst = await this.deps.api.showInstance(id).catch(() => null)
    const base: VastInstance = inst ?? {
      id,
      label: intent.label,
      status: 'created',
      statusMessage: '',
      gpuName: '',
      numGpus: 1,
      gpuRamMiB: 0,
      dollarsPerHour: intent.dollarsPerHour,
      storageDollarsPerHour: null,
      diskGb: intent.diskGb,
      startedAt: null,
      image: this.settings.image
    }
    const rec = this.blankRec({ ...base, label: intent.label, dollarsPerHour: base.dollarsPerHour || intent.dollarsPerHour }, 'app', intent.autoStop)
    rec.rentalStatus = 'starting'
    rec.awaitingSince = this.now()
    rec.pendingModelId = intent.modelId
    rec.agreedPrice = intent.dollarsPerHour
    this.recs.set(id, rec)
    this.beginPeriod(rec, this.now())
    this.note(id, `Rented offer ${intent.offerId} at $${intent.dollarsPerHour.toFixed(3)} per hour.`)
    if (inst) {
      rec.routes = inst.routes
      this.applyInstance(rec, { ...inst, label: intent.label })
    }
    this.intents = this.intents.filter((i) => i !== intent)
    this.intentRt.delete(intent.label)
    await this.saveNow().catch((e) => console.error('Could not save remote.json:', e))
    this.emit()
  }

  // ---------- intents ----------

  private async settleIntents(): Promise<void> {
    for (const intent of [...this.intents]) {
      if (intent.state !== 'checking') continue
      const rt = this.intentRt.get(intent.label) ?? { since: this.now(), done: 0 }
      this.intentRt.set(intent.label, rt)
      if (this.now() < rt.since + CHECK_OFFSETS[rt.done] * 1000) continue
      // A search that failed says nothing about whether it was rented, so it is not one of the checks.
      if ((await this.lookForIntent(intent, false)) === null) continue
      rt.done++
      if (this.intents.includes(intent) && rt.done >= CHECK_OFFSETS.length) {
        intent.state = 'unresolved'
        intent.message = `No instance labelled ${intent.label} was found in 10 minutes, but one may still exist. Look on vast.ai's site, then check again or dismiss this.`
        this.save()
        this.emit()
      }
    }
  }

  /** One search of the whole account for the label. True: a rental was adopted; null: the account could not be read. */
  private async lookForIntent(intent: IntentRec, manual: boolean): Promise<boolean | null> {
    let list: VastInstanceDetail[]
    try {
      list = await this.deps.api.listInstances()
    } catch (e) {
      if (manual) throw new Error(`Could not read your vast.ai instances. ${this.msg(e)}`)
      return null
    }
    const hit = list.find((i) => i.label === intent.label)
    if (!hit) return false
    await this.adoptContract(intent, hit.id, hit)
    return true
  }

  async checkIntent(label: string): Promise<RemoteState> {
    const intent = this.intents.find((i) => i.label === label)
    if (!intent) return this.state()
    if (intent.state === 'sending') throw new Error('The rent request is still being sent.')
    const found = await this.lookForIntent(intent, true)
    if (!found) {
      intent.message =
        intent.state === 'unresolved'
          ? `Still no instance labelled ${intent.label}. Look on vast.ai's site, then dismiss this if there is none.`
          : intent.message
      this.emit()
    }
    return this.state()
  }

  async dismissIntent(label: string): Promise<RemoteState> {
    const intent = this.intents.find((i) => i.label === label)
    if (!intent) return this.state()
    // A restart begins the checks again; an old request that was looked for once may still be dismissed.
    const old = this.now() - intent.createdAt >= CHECK_OFFSETS[CHECK_OFFSETS.length - 1] * 1000 && (this.intentRt.get(label)?.done ?? 0) > 0
    if (intent.state !== 'unresolved' && !old) throw new Error('The app is still checking for this rent. It can be dismissed after the checks end.')
    this.intents = this.intents.filter((i) => i !== intent)
    this.intentRt.delete(label)
    await this.saveNow()
    this.emit()
    return this.state()
  }

  // ---------- attach, detach ----------

  async attach(instanceId: number): Promise<RemoteState> {
    if (!Number.isSafeInteger(instanceId) || instanceId <= 0) throw new Error('That instance id is not valid.')
    if (this.recs.has(instanceId)) throw new Error('That instance is already in the list.')
    const list = await this.deps.api.listInstances()
    const inst = list.find((i) => i.id === instanceId)
    if (!inst) throw new Error('That instance is not on your vast.ai account.')
    const rec = this.blankRec({ ...inst }, 'attached', { ...ATTACHED_AUTO_STOP })
    this.recs.set(instanceId, rec)
    this.applyInstance(rec, inst)
    this.note(instanceId, 'Attached. The app changes nothing on it except its own folder and, when auto-stop is switched on, the watchdog.')
    await this.saveNow()
    this.emit()
    return this.state()
  }

  async detach(rentalId: number): Promise<RemoteState> {
    const rec = this.recs.get(rentalId)
    if (!rec) return this.state()
    if (rec.origin === 'app' && rec.rentalStatus !== 'gone') {
      throw new Error('A rental the app made stays in the list: stop it, or destroy it, instead.')
    }
    if (rec.uses.size) throw new Error(`It is in use (${[...rec.uses.values()].map((u) => u.what).join(', ')}).`)
    rec.setup?.abort.abort()
    this.deps.ops.disconnect(rentalId)
    this.recs.delete(rentalId)
    await this.saveNow()
    this.emit()
    return this.state()
  }

  // ---------- start, stop, destroy ----------

  private rec(id: number): Rec {
    if (!Number.isSafeInteger(id)) throw new Error('That rental is not valid.')
    const r = this.recs.get(id)
    if (!r) throw new Error('The app does not use that instance.')
    return r
  }

  /** Re-reads the instance by id right before acting on it. */
  private async reread(rec: Rec): Promise<VastInstanceDetail> {
    let inst: VastInstanceDetail | null
    try {
      inst = await this.deps.api.showInstance(rec.inst.id)
    } catch (e) {
      throw new Error(`Could not read the instance from vast.ai, so nothing was changed. ${this.msg(e)}`)
    }
    if (!inst) throw new Error('vast.ai does not show that instance. Nothing was changed; check vast.ai and remove it from the list if it is gone.')
    this.applyInstance(rec, inst)
    return inst
  }

  private useList(rec: Rec): string {
    return [...rec.uses.values()].map((u) => u.what).join(', ')
  }

  /** Bounded so a stuck unload cannot keep the instance billing. */
  private async runStopping(rentalId: number): Promise<void> {
    const hook = this.deps.onStopping
    if (!hook) return
    let timer: ReturnType<typeof setTimeout> | undefined
    try {
      await Promise.race([
        hook(rentalId).catch((e) => console.error('onStopping failed:', this.msg(e))),
        new Promise<void>((resolve) => {
          timer = setTimeout(resolve, this.deps.stopHookTimeoutMs ?? 30_000)
        })
      ])
    } finally {
      clearTimeout(timer)
    }
  }

  async start(rentalId: number): Promise<RemoteState> {
    const rec = this.rec(rentalId)
    await this.reread(rec)
    if (rec.rentalStatus !== 'stopped') throw new Error('Only a stopped instance can be started.')
    await this.deps.api.startInstance(rentalId).catch((e) => {
      throw new Error(this.msg(e))
    })
    const now = this.now()
    rec.rentalStatus = 'starting'
    rec.awaitingSince = now
    rec.stopRequestedAt = null
    rec.error = ''
    rec.countdown = null
    rec.suppressUntil = 0
    rec.ssh = { state: 'unknown' }
    this.beginPeriod(rec, now)
    this.note(rentalId, 'Start requested.')
    this.save()
    this.emit()
    return this.state()
  }

  async stop(rentalId: number, force: boolean): Promise<RemoteState> {
    const rec = this.rec(rentalId)
    await this.reread(rec)
    if (rec.rentalStatus === 'stopped' || rec.rentalStatus === 'stopping') return this.state()
    if (rec.uses.size && !force) throw new Error(`It is in use (${this.useList(rec)}). Stop it anyway to cancel that use.`)
    if (force) await this.runStopping(rentalId)
    await this.deps.api.stopInstance(rentalId).catch((e) => {
      throw new Error(this.msg(e))
    })
    this.afterStopSent(rec, 'Stop requested.')
    return this.state()
  }

  private afterStopSent(rec: Rec, why: string): void {
    this.accrue(rec)
    rec.stopRequestedAt = this.now()
    rec.rentalStatus = 'stopping'
    rec.countdown = null
    rec.error = ''
    rec.awaitingSince = null
    this.deps.ops.disconnect(rec.inst.id)
    rec.ssh = { state: 'unknown' }
    this.note(rec.inst.id, why)
    this.save()
    this.emit()
  }

  async destroy(rentalId: number, label: string, force: boolean): Promise<RemoteState> {
    const rec = this.rec(rentalId)
    if (rec.origin !== 'app') throw new Error('Only instances the app rented can be destroyed here. Do it on vast.ai for this one.')
    if (typeof label !== 'string' || label !== rec.inst.label) throw new Error('The name given does not match this instance, so nothing was destroyed.')
    const inst = await this.reread(rec)
    if (inst.label !== label) throw new Error('The label on vast.ai is not the one the app made, so nothing was destroyed.')
    if (rec.uses.size && !force) throw new Error(`It is in use (${this.useList(rec)}). Destroy it anyway to cancel that use.`)
    if (force) await this.runStopping(rentalId)
    await this.deps.api.destroyInstance(rentalId).catch((e) => {
      throw new Error(this.msg(e))
    })
    rec.setup?.abort.abort()
    this.deps.ops.disconnect(rentalId)
    rec.destroySent = true
    // Dropped from the list only when vast.ai no longer shows it; until then it may still cost money.
    const left = await this.deps.api.showInstance(rentalId).catch(() => rec.inst)
    if (left) {
      rec.error = 'The destroy was sent, but vast.ai still shows the instance. It is billed until it disappears here; if it stays, look on vast.ai.'
      this.note(rentalId, 'Destroy sent.')
    } else this.recs.delete(rentalId)
    await this.saveNow().catch((e) => console.error('Could not save remote.json:', e))
    this.emit()
    return this.state()
  }

  // ---------- auto-stop ----------

  async setAutoStop(rentalId: number, patch: Partial<AutoStop>): Promise<RemoteState> {
    const rec = this.rec(rentalId)
    const next = cleanAutoStop(patch, rec.autoStop)
    const capChanged = next.capHours !== rec.autoStop.capHours
    // Idle time counts from when the idle stop was switched on, not from before anyone asked for it.
    if (next.enabled && !rec.autoStop.enabled) rec.lastUsedAt = this.now()
    rec.autoStop = next
    if (capChanged) {
      // An attached instance may have been running for days: its limit counts from the moment it is switched on.
      const from = rec.origin === 'attached' ? this.now() : rec.periodStart
      rec.capAt = next.capHours !== null && from !== null ? from + next.capHours * HOUR : null
    }
    this.dropStaleCountdown(rec)
    rec.watchdog = { state: 'unknown' }
    await this.saveNow()
    this.emit()
    if (rec.rentalStatus === 'running' && rec.ssh.state === 'ok') this.spawn(() => this.armWatchdog(rec))
    return this.state()
  }

  private dropStaleCountdown(rec: Rec): void {
    const c = rec.countdown
    if (!c) return
    const now = this.now()
    if (c.kind === 'idle' && !rec.autoStop.enabled) rec.countdown = null
    if (c.kind === 'cap' && (rec.autoStop.capHours === null || rec.capAt === null || rec.capAt > now)) rec.countdown = null
  }

  async keepRunning(rentalId: number): Promise<RemoteState> {
    const rec = this.rec(rentalId)
    const c = rec.countdown
    if (c) {
      if (c.kind === 'cap') rec.capAt = Math.max(rec.capAt ?? 0, this.now()) + HOUR
      else rec.lastUsedAt = this.now()
      rec.countdown = null
      rec.suppressUntil = 0
      if (rec.error.startsWith('Could not stop it')) rec.error = ''
      this.rearm(rec)
      this.note(rentalId, 'Keep running chosen.')
      this.save()
      this.emit()
    }
    return this.state()
  }

  async extendCap(rentalId: number, hours: number): Promise<RemoteState> {
    const rec = this.rec(rentalId)
    if (!Number.isInteger(hours) || hours < 1 || hours > 720) throw new Error('Extend by a whole number of hours from 1 to 720.')
    if (rec.autoStop.capHours === null) throw new Error('This instance has no hour limit to extend. Set one first.')
    rec.capAt = Math.max(rec.capAt ?? 0, this.now()) + hours * HOUR
    this.dropStaleCountdown(rec)
    this.rearm(rec)
    this.save()
    this.emit()
    return this.state()
  }

  /** The watchdog on the instance has its own copy of the limits: it gets the new ones (and a fresh keepalive). */
  private rearm(rec: Rec): void {
    // Not "armed" until it has them: the quit question would promise the wrong time.
    rec.watchdog = { state: 'unknown' }
    if (rec.rentalStatus === 'running' && rec.ssh.state === 'ok') this.spawn(() => this.armWatchdog(rec))
  }

  /** A batch run ended: stop these rentals after the usual countdown, unless something else needs them. */
  stopAfterRun(rentalIds: number[]): void {
    const now = this.now()
    for (const id of rentalIds) {
      const rec = this.recs.get(id)
      if (!rec || rec.rentalStatus !== 'running') continue
      if (rec.uses.size) {
        this.note(id, `Not stopping after the batch run: it is in use (${this.useList(rec)}).`)
        continue
      }
      if (rec.foreignProcesses) {
        this.note(id, 'Not stopping after the batch run: another process is using its GPU.')
        continue
      }
      if (!rec.countdown) rec.countdown = { at: now + STOP_COUNTDOWN_SECONDS * 1000, reason: 'The batch run ended', kind: 'batch' }
    }
    this.emit()
  }

  // ---------- watchdog ----------

  private armWatchdog(rec: Rec): Promise<void> {
    const seq = ++rec.armSeq
    const run = rec.arming.then(() => (seq === rec.armSeq ? this.armNow(rec, seq) : undefined))
    rec.arming = run.catch(() => undefined)
    return run
  }

  private async armNow(rec: Rec, seq: number): Promise<void> {
    const id = rec.inst.id
    const a = rec.autoStop
    const wanted = rec.origin === 'app' || a.enabled || a.capHours !== null || rec.armedByProbe
    if (!wanted) {
      rec.watchdog = { state: 'not-armed' }
      this.emit()
      return
    }
    try {
      const res = await this.deps.ops.armWatchdog(id, { autoStop: { ...a }, capAt: a.capHours !== null ? rec.capAt : null, origin: rec.origin })
      // Newer limits are waiting to be sent: these are not the ones that count.
      if (seq !== rec.armSeq) return
      rec.watchdog = res.armed ? { state: 'armed' } : { state: 'not-armed', ...(res.error ? { error: this.scrub(res.error) } : {}) }
    } catch (e) {
      rec.watchdog = { state: 'not-armed', error: this.msg(e) }
    }
    this.emit()
  }

  // ---------- bring-up and setup ----------

  private async bringUp(rec: Rec): Promise<void> {
    const id = rec.inst.id
    rec.bringingUp = true
    rec.ssh = { state: 'connecting' }
    this.emit()
    try {
      try {
        this.applyProbe(rec, await this.deps.ops.probe(id))
      } catch (e) {
        rec.ssh = { state: 'failed', error: this.msg(e) }
        rec.nextProbeAt = this.now() + PROBE_RETRY_MS
        this.emit()
        return
      }
      this.emit()
      await this.armWatchdog(rec)
      if (rec.pendingModelId && rec.rentalStatus === 'running' && !rec.setup) {
        const def = this.deps.models.list().find((m) => m.id === rec.pendingModelId)
        if (def) this.startSetup(rec, def)
        else rec.pendingModelId = ''
      }
    } finally {
      rec.bringingUp = false
    }
  }

  async setup(rentalId: number, modelId: string | null): Promise<RemoteState> {
    const rec = this.rec(rentalId)
    if (rec.rentalStatus !== 'running') throw new Error('The instance must be running before it can be set up.')
    if (rec.setup) throw new Error('A setup is already running on this instance.')
    let def: RemoteModelDef | null = null
    if (modelId !== null) {
      def = this.deps.models.list().find((m) => m.id === modelId) ?? null
      if (!def) throw new Error('That remote model is not saved.')
    }
    this.startSetup(rec, def)
    return this.state()
  }

  private startSetup(rec: Rec, def: RemoteModelDef | null): void {
    const abort = new AbortController()
    rec.setup = { abort, modelId: def?.id ?? null }
    this.spawn(() => this.runSetup(rec, def, abort))
  }

  async cancelSetup(rentalId: number): Promise<RemoteState> {
    const rec = this.rec(rentalId)
    rec.setup?.abort.abort()
    return this.state()
  }

  private progress(rec: Rec, entry: RentalModel | null, p: TransferProgress): void {
    if (entry) {
      entry.stage = p.stage
      entry.bytesDone = p.bytesDone
      entry.bytesTotal = p.bytesTotal
    } else {
      rec.engine.stage = p.stage
    }
    const now = this.now()
    if (now - this.lastProgressEmit >= PROGRESS_EVERY_MS) {
      this.lastProgressEmit = now
      this.emit()
    }
  }

  private async runSetup(rec: Rec, def: RemoteModelDef | null, abort: AbortController): Promise<void> {
    const id = rec.inst.id
    const release = this.use(id, 'setup', 'window')
    const tag = this.tag()
    let entry: RentalModel | null = null
    try {
      let probe: ProbeResult
      try {
        probe = await this.deps.ops.probe(id, abort.signal)
        this.applyProbe(rec, probe)
      } catch (e) {
        throw new Error(`Could not reach the instance: ${this.msg(e)}`)
      }
      // applyProbe leaves engine and model states alone while a setup runs, so read the engine from the probe itself.
      const installed = probe.engines.includes(tag)
      const missing = def ? await this.deps.ops.missingBytes(id, def) : 0
      const free = rec.diskFreeBytes
      if (free !== null && missing !== null) {
        const need = missing + KEEP_FREE_BYTES + (installed ? 0 : ENGINE_BYTES)
        if (need > free) {
          const text = `Not enough free disk: about ${(need / 1e9).toFixed(1)} GB are needed (files, the engine and 5 GB spare) and ${(free / 1e9).toFixed(1)} GB are free.`
          if (rec.origin === 'attached') throw new Error(`${text} Nothing was installed on this instance.`)
          this.note(id, `Warning: ${text} Going on anyway.`)
        }
      } else if (missing === null) this.note(id, 'The size of the model files is not known, so free disk was not checked.')
      if (!installed) {
        rec.engine = { state: 'working', tag, stage: 'Installing the engine' }
        this.emit()
        await this.deps.ops.installEngine(id, tag, (p) => this.progress(rec, null, p), abort.signal)
        rec.engine = { state: 'ready', tag, stage: '' }
      }
      if (def) {
        entry = rec.models.find((m) => m.modelId === def.id) ?? null
        if (!entry) {
          entry = { modelId: def.id, state: 'working', bytesDone: 0, bytesTotal: 0, stage: 'Starting' }
          rec.models.push(entry)
        } else Object.assign(entry, { state: 'working', stage: 'Starting', error: undefined })
        this.emit()
        await this.deps.ops.transferModel(id, def, (p) => this.progress(rec, entry, p), abort.signal)
        entry.state = 'ready'
        entry.stage = ''
        entry.bytesDone = entry.bytesTotal
      }
      rec.pendingModelId = ''
      this.note(id, def ? `Set up ${def.name}.` : 'The engine is ready.')
    } catch (e) {
      const text = abort.signal.aborted ? 'Cancelled.' : this.msg(e)
      this.note(id, `Setup stopped: ${text}`)
      if (rec.engine.state === 'working') rec.engine = { ...rec.engine, state: 'failed', stage: '', error: text }
      if (def && !entry) {
        entry = rec.models.find((m) => m.modelId === def.id) ?? null
        if (!entry) rec.models.push((entry = { modelId: def.id, state: 'working', bytesDone: 0, bytesTotal: 0, stage: '' }))
      }
      if (entry && entry.state === 'working') Object.assign(entry, { state: 'failed', stage: '', error: text })
      // A failed setup is not retried by itself after a restart: the user sees the failure and decides.
      rec.pendingModelId = ''
    } finally {
      release()
      rec.setup = null
      this.save()
      this.emit()
    }
  }

  async removeModelFiles(rentalId: number, modelId: string): Promise<RemoteState> {
    const rec = this.rec(rentalId)
    if (!isRemoteModelId(modelId)) throw new Error('That is not a remote model.')
    if (rec.rentalStatus !== 'running') throw new Error('The instance must be running to remove files from it.')
    if (rec.setup) throw new Error('A setup is running on this instance. Cancel it first.')
    await this.deps.ops.removeModelFiles(rentalId, modelId)
    rec.models = rec.models.filter((m) => m.modelId !== modelId)
    this.emit()
    return this.state()
  }

  // ---------- tick ----------

  /** Called every 10 s (startTicking) and by tests with their own clock. Never overlaps itself. */
  tick(): Promise<void> {
    if (this.ticking) return this.ticking
    const p = this.tickBody().finally(() => {
      this.ticking = null
    })
    this.ticking = p
    return p
  }

  startTicking(): void {
    if (this.timer !== null || this.disposed) return
    const set = this.deps.setTimer ?? ((fn, ms) => setInterval(fn, ms))
    this.timer = set(() => {
      void this.tick().catch((e) => console.error('Remote tick failed:', this.msg(e)))
    }, 10_000)
  }

  /** Stops the timers and drops every connection. Synchronous: the app is closing. */
  dispose(): void {
    this.disposed = true
    if (this.timer !== null) {
      const clear = this.deps.clearTimer ?? ((h) => clearInterval(h as ReturnType<typeof setInterval>))
      clear(this.timer)
      this.timer = null
    }
    for (const [id, rec] of this.recs) {
      rec.setup?.abort.abort()
      try {
        this.deps.ops.disconnect(id)
      } catch (e) {
        console.error('disconnect failed:', e)
      }
    }
  }

  private async tickBody(): Promise<void> {
    if (this.disposed) return
    await this.settleIntents()
    if (this.listDue()) await this.refreshList()
    const now = this.now()
    for (const [id, rec] of this.recs) {
      this.accrue(rec)
      if (rec.rentalStatus === 'gone') continue
      if (this.countedUses(rec) > 0) rec.lastUsedAt = now
      if (rec.rentalStatus === 'running') {
        if (rec.ssh.state !== 'ok' && now >= rec.nextProbeAt && !rec.bringingUp) {
          rec.nextProbeAt = now + PROBE_RETRY_MS
          this.spawn(() => this.bringUp(rec))
        } else if (rec.ssh.state === 'ok' && !rec.setup && !rec.bringingUp && now - rec.lastProbeAt >= PROBE_EVERY_MS) {
          rec.lastProbeAt = now
          this.spawn(async () => {
            try {
              this.applyProbe(rec, await this.deps.ops.probe(id))
            } catch (e) {
              rec.ssh = { state: 'failed', error: this.msg(e) }
            }
            this.emit()
          })
        }
        await this.checkLimits(rec)
      }
    }
    this.emit()
  }

  private async checkLimits(rec: Rec): Promise<void> {
    const now = this.now()
    const counted = this.countedUses(rec)
    const a = rec.autoStop
    if (rec.countdown) {
      this.dropStaleCountdown(rec)
      if (rec.countdown && rec.countdown.kind === 'idle' && counted > 0) rec.countdown = null
      if (rec.countdown && now >= rec.countdown.at && !rec.finalizing) await this.finalize(rec)
      return
    }
    if (now < rec.suppressUntil) return
    const capDue = a.capHours !== null && rec.capAt !== null && now >= rec.capAt
    const idleDue = a.enabled && counted === 0 && now >= (rec.lastUsedAt ?? rec.periodStart ?? now) + a.idleMinutes * MINUTE
    const at = now + STOP_COUNTDOWN_SECONDS * 1000
    if (capDue) rec.countdown = { at, kind: 'cap', reason: `The time limit of ${a.capHours} hours is reached.` }
    else if (idleDue) rec.countdown = { at, kind: 'idle', reason: `It has been idle for ${a.idleMinutes} minutes.` }
    if (rec.countdown) this.note(rec.inst.id, `${rec.countdown.reason} It will stop in ${STOP_COUNTDOWN_SECONDS} seconds unless you choose Keep running.`)
  }

  /** The countdown ended: check again before stopping, with a fresh look at the GPUs. */
  private async finalize(rec: Rec): Promise<void> {
    const cd = rec.countdown
    if (!cd) return
    const id = rec.inst.id
    rec.finalizing = true
    try {
      const veto = (why: string): void => {
        if (rec.countdown === cd) rec.countdown = null
        // Keep running and a new use have set their own times; only a standing reason needs the pause.
        if (why !== KEPT) rec.suppressUntil = this.now() + VETO_MS
        rec.error = rec.error.startsWith('Could not stop it') ? '' : rec.error
        this.note(id, `Not stopping: ${why}`)
      }
      const first = this.notDue(rec, cd)
      if (first) return veto(first)
      let probed = true
      try {
        this.applyProbe(rec, await this.deps.ops.probe(id))
      } catch {
        probed = false
      }
      if (probed && rec.foreignProcesses && (cd.kind !== 'cap' || rec.origin === 'attached')) {
        return veto('another process is using its GPU.')
      }
      // Work on the GPU while nothing here uses the rental is somebody else's, even if its process is not visible.
      if (probed && cd.kind !== 'cap' && rec.uses.size === 0 && rec.gpus.some((g) => g.utilization > BUSY_PERCENT)) {
        return veto('its GPU is busy although nothing here uses it.')
      }
      if (!probed && rec.origin !== 'app') return veto('the GPUs could not be checked and this instance was not rented by the app.')
      if (!probed && rec.foreignProcesses && cd.kind !== 'cap') return veto('another process was using its GPU when it was last reached.')
      await this.autoStop(rec, cd, veto)
    } finally {
      rec.finalizing = false
    }
  }

  /** Why a countdown that ended must not stop the rental after all; null = it may. */
  private notDue(rec: Rec, cd: NonNullable<Rec['countdown']>): string | null {
    const now = this.now()
    // Keep running, or a use that ends this kind of countdown, took it away while something was awaited.
    if (rec.countdown !== cd) return KEPT
    if (cd.kind === 'idle') {
      if (this.countedUses(rec) > 0) return 'it is in use again.'
      if (!rec.autoStop.enabled) return 'auto-stop was switched off.'
      if (now < (rec.lastUsedAt ?? rec.periodStart ?? now) + rec.autoStop.idleMinutes * MINUTE) return 'it was used a moment ago.'
    } else if (cd.kind === 'cap') {
      if (rec.autoStop.capHours === null || rec.capAt === null || rec.capAt > now) return 'the time limit was changed.'
    } else if (rec.uses.size) return `it is in use (${this.useList(rec)}).`
    return null
  }

  /** The app's own stop. A failure is shown and tried again after another countdown, with every check made again. */
  private async autoStop(rec: Rec, cd: NonNullable<Rec['countdown']>, veto: (why: string) => void): Promise<void> {
    const id = rec.inst.id
    try {
      const inst = await this.reread(rec)
      if (inst.status !== 'running') {
        rec.countdown = null
        return
      }
      const before = this.notDue(rec, cd)
      if (before) return veto(before)
      await this.runStopping(id)
      const after = this.notDue(rec, cd)
      if (after) return veto(after)
      await this.deps.api.stopInstance(id)
      this.afterStopSent(rec, `Stopped by the app. ${cd.reason}`)
    } catch (e) {
      if (rec.countdown !== cd) return
      rec.countdown = { ...cd, at: this.now() + MINUTE }
      rec.error = `Could not stop it: ${this.msg(e)} Trying again in a minute.`
      this.emit()
    }
  }

  // ---------- quit ----------

  quitSummary(): { running: { label: string; dollarsPerHour: number; stopsItself: string | null }[] } {
    const at = (ms: number): string => new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    const running = [...this.recs.values()]
      .filter((r) => r.rentalStatus === 'running' || r.rentalStatus === 'starting')
      .map((r) => {
        const a = r.autoStop
        let stops: string | null = null
        // Only the watchdog on the instance works with the app closed.
        if (r.watchdog.state === 'armed' && (a.enabled || a.capHours !== null)) {
          const parts: string[] = []
          if (a.enabled) parts.push(`after ${a.idleMinutes} idle minutes`)
          if (a.capHours !== null && r.capAt !== null) parts.push(`at the latest at ${at(r.capAt)}`)
          stops = parts.join(', ')
        }
        return { label: r.inst.label || `Instance ${r.inst.id}`, dollarsPerHour: r.inst.dollarsPerHour, stopsItself: stops }
      })
    return { running }
  }

  /** "Stop and quit": stops every running rental. Returns what could not be stopped. */
  async stopAllForQuit(): Promise<string[]> {
    const errors: string[] = []
    for (const [id, rec] of this.recs) {
      if (rec.rentalStatus !== 'running' && rec.rentalStatus !== 'starting') continue
      const name = rec.inst.label || `Instance ${id}`
      try {
        const inst = await this.reread(rec)
        if (inst.status !== 'running' && rec.rentalStatus !== 'starting') continue
        if (rec.origin === 'attached') {
          // Rented elsewhere, maybe training: only a fresh look says what runs there now.
          const p = await this.deps.ops.probe(id).catch(() => null)
          if (!p || p.foreignProcesses) {
            errors.push(`${name} was left running: ${p ? 'another process is using its GPU' : 'its GPU could not be checked, and it was not rented by the app'}.`)
            continue
          }
        }
        await this.runStopping(id)
        await this.deps.api.stopInstance(id)
        this.afterStopSent(rec, 'Stopped when the app was closed.')
      } catch (e) {
        errors.push(`${name}: ${this.msg(e)}`)
      }
    }
    return errors
  }
}
