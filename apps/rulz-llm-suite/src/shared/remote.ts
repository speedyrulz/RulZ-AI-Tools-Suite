// Models on rented vast.ai instances: types, pure helpers and the IPC contract (docs/vast-design.md).
import type { LoadConfig, LoadedInstance, MemoryEstimate } from './types'

/** Ids of remote models start with this; library ids are relative paths and never do. */
export const REMOTE_MODEL_PREFIX = 'vast/'
export const isRemoteModelId = (id: string | null | undefined): boolean => !!id && id.startsWith(REMOTE_MODEL_PREFIX)
export const REMOTE_SLUG = /^[a-z0-9][a-z0-9._-]{0,59}$/

/** Where an instance runs; absent on LoadedInstance = this PC. */
export interface RemoteHost {
  kind: 'vast'
  rentalId: number
  label: string
}

export const DEFAULT_ENGINE_TAG = 'b12407'
export const MAX_SESSIONS_PER_HOST = 8
export const MAX_LANES = 16
/** Seconds between a decision to stop by the app's own timer and the stop. */
export const STOP_COUNTDOWN_SECONDS = 60

// ---------- Remote models ----------

export type RemoteModelSource =
  /** A model in the local library, uploaded with its parts, projector, draft and adapters. */
  | { kind: 'local'; modelId: string }
  /** `repo` is "owner/name" of a model repo, or of a bucket when `bucket` is set. `file` may be the first part of a split GGUF. */
  | { kind: 'hf'; repo: string; bucket: boolean; revision: string; file: string; mmproj: string }

export interface RemoteModelDef {
  /** REMOTE_MODEL_PREFIX + slug. */
  id: string
  name: string
  source: RemoteModelSource
  createdAt: number
}

// ---------- Offers ----------

export interface OfferFilter {
  /** GB of VRAM over all GPUs of the offer. */
  minVramGb: number
  /** '' = any. */
  gpuName: string
  maxGpus: number
  maxDollarsPerHour: number | null
  diskGb: number
  /** 0..1 */
  minReliability: number
  minInetDownMbps: number
}

export const DEFAULT_OFFER_FILTER: OfferFilter = {
  minVramGb: 24,
  gpuName: '',
  maxGpus: 1,
  maxDollarsPerHour: 1,
  diskGb: 60,
  minReliability: 0.98,
  minInetDownMbps: 200
}

export interface RemoteOffer {
  id: number
  machineId: number
  gpuName: string
  numGpus: number
  /** Per GPU. */
  gpuRamMiB: number
  cpuCores: number
  cpuRamMiB: number
  /** Including the disk size asked for. */
  dollarsPerHour: number
  reliability: number
  inetDownMbps: number
  inetUpMbps: number
  /** Price of a GB received / sent by the instance; null = not reported. */
  inetDownPerGb: number | null
  inetUpPerGb: number | null
  cudaMaxGood: number
  geolocation: string
  /** Longest the host rents it out, in days; null = not reported. */
  maxDays: number | null
}

// ---------- Rentals ----------

export interface AutoStop {
  enabled: boolean
  idleMinutes: number
  /** Hours after start at which it stops whatever runs; null = no cap. */
  capHours: number | null
  /** Requests from the API server and the phone keep it running too. */
  apiCountsAsUse: boolean
}

export const DEFAULT_AUTO_STOP: AutoStop = { enabled: true, idleMinutes: 30, capHours: 12, apiCountsAsUse: false }
export const ATTACHED_AUTO_STOP: AutoStop = { enabled: false, idleMinutes: 30, capHours: null, apiCountsAsUse: false }

/** An instance of the account, as vast.ai lists it. */
export interface VastInstance {
  id: number
  label: string
  /** vast's actual_status: running, loading, exited, offline, ... */
  status: string
  statusMessage: string
  gpuName: string
  numGpus: number
  gpuRamMiB: number
  dollarsPerHour: number
  /** What it costs while stopped (its disk); null = not reported. */
  storageDollarsPerHour: number | null
  diskGb: number
  /** Epoch ms; null = not reported. */
  startedAt: number | null
  image: string
}

export type RentalStatus = 'starting' | 'running' | 'stopping' | 'stopped' | 'gone' | 'unknown'
export type SetupState = 'none' | 'working' | 'ready' | 'failed'

export interface RemoteGpu {
  index: number
  name: string
  totalMiB: number
  usedMiB: number
  /** Percent. */
  utilization: number
}

export interface RentalModel {
  modelId: string
  state: SetupState
  bytesDone: number
  bytesTotal: number
  /** 'Downloading part 1 of 3', 'Checking', ... */
  stage: string
  error?: string
}

/** An instance the app uses: rented by it, or attached. */
export interface Rental extends VastInstance {
  origin: 'app' | 'attached'
  rentalStatus: RentalStatus
  ssh: { state: 'unknown' | 'connecting' | 'ok' | 'failed'; error?: string }
  engine: { state: SetupState; tag: string; stage: string; error?: string }
  models: RentalModel[]
  gpus: RemoteGpu[]
  diskFreeBytes: number | null
  /** A compute process that is not this app's llama-server is on a GPU. */
  foreignProcesses: boolean
  autoStop: AutoStop
  watchdog: { state: 'armed' | 'not-armed' | 'unknown'; error?: string }
  /** What is using it now ("batch run", "setup", "chat"); empty = idle. */
  inUse: string[]
  /** When it last was in use (epoch ms). */
  lastUsedAt: number | null
  /** When the idle limit / the hard cap will stop it; null = not set or in use. */
  idleStopAt: number | null
  capAt: number | null
  /** The app is about to stop it: when and why. */
  countdown: { at: number; reason: string } | null
  /** Estimated cost of this running period so far. */
  spent: number
  /** What went wrong with it last. */
  error?: string
}

/** A rent request whose outcome is not known yet. */
export interface RentIntent {
  /** The unique label sent with the request. */
  label: string
  offerId: number
  dollarsPerHour: number
  createdAt: number
  state: 'sending' | 'checking' | 'unresolved'
  message: string
}

export interface RemoteSettings {
  /** '' = the same build as this PC's llama-server. */
  engineTag: string
  image: string
}

export const DEFAULT_REMOTE_SETTINGS: RemoteSettings = { engineTag: '', image: 'vastai/base-image:@vastai-automatic-tag' }

export interface RemoteState {
  /** Goes up with every state sent; a lower one is older. */
  rev: number
  rentals: Rental[]
  intents: RentIntent[]
  models: RemoteModelDef[]
  settings: RemoteSettings
  /** Where the key or token comes from, '' = none. Never the value. */
  vastKeySource: string
  hfTokenSource: string
  /** Account balance in dollars; null = not read. */
  credit: number | null
  ssh: { available: boolean; publicKeyPath: string | null; onAccount: boolean | null; error?: string }
  /** The engine tag a setup would install, resolved from the settings and this PC's llama-server. */
  resolvedEngineTag: string
  /** Last error talking to vast.ai. */
  error: string | null
}

export interface RentRequest {
  offerId: number
  /** The price the user agreed to. */
  shownDollarsPerHour: number
  filter: OfferFilter
  autoStop: AutoStop
  /** Set it up for this remote model once it runs; '' = not yet. */
  modelId: string
}

// ---------- Batch targets ----------

export type BatchHost = { kind: 'local' } | { kind: 'vast'; rentalId: number }

/** Where a batch run runs. */
export interface BatchTarget {
  hosts: BatchHost[]
  /** Tasks at once on each host. */
  sessionsPerHost: number
  /** Context of each session; null = the model's setting (one session only). */
  contextPerSession: number | null
  /** Stop the rentals of this run when it ends (after a countdown). */
  stopRentalsAfter: boolean
}

export const DEFAULT_BATCH_TARGET: BatchTarget = { hosts: [{ kind: 'local' }], sessionsPerHost: 1, contextPerSession: null, stopRentalsAfter: false }

/** A target from disk or the window with anything unusable replaced; duplicates of a host dropped. */
export function cleanBatchTarget(raw: unknown): BatchTarget {
  const d = DEFAULT_BATCH_TARGET
  if (!raw || typeof raw !== 'object') return { ...d, hosts: [...d.hosts] }
  const r = raw as Record<string, unknown>
  const hosts: BatchHost[] = []
  for (const h of Array.isArray(r.hosts) ? r.hosts : []) {
    const x = h as { kind?: unknown; rentalId?: unknown } | null
    if (x?.kind === 'local' && !hosts.some((y) => y.kind === 'local')) hosts.push({ kind: 'local' })
    else if (x?.kind === 'vast' && Number.isSafeInteger(x.rentalId) && !hosts.some((y) => y.kind === 'vast' && y.rentalId === x.rentalId)) {
      hosts.push({ kind: 'vast', rentalId: x.rentalId as number })
    }
  }
  if (!hosts.length) hosts.push({ kind: 'local' })
  const most = Math.max(1, Math.min(MAX_SESSIONS_PER_HOST, Math.floor(MAX_LANES / hosts.length)))
  const n = typeof r.sessionsPerHost === 'number' && Number.isFinite(r.sessionsPerHost) ? Math.round(r.sessionsPerHost) : d.sessionsPerHost
  const ctx = typeof r.contextPerSession === 'number' && Number.isFinite(r.contextPerSession) && r.contextPerSession >= 1024 ? Math.round(r.contextPerSession) : null
  return {
    hosts: hosts.slice(0, MAX_LANES),
    sessionsPerHost: Math.min(most, Math.max(1, n)),
    contextPerSession: ctx,
    stopRentalsAfter: r.stopRentalsAfter === true
  }
}

// ---------- IPC ----------

export interface RemoteIpcInvoke {
  'remote:state': () => RemoteState
  /** Re-reads instances, credit and connections now. */
  'remote:refresh': () => RemoteState
  /** Saves (encrypted) or with '' forgets the key typed in the app. */
  'remote:setVastKey': (key: string) => RemoteState
  'remote:setHfToken': (token: string) => RemoteState
  'remote:saveSettings': (settings: RemoteSettings) => RemoteState
  /** Adds this PC's public ssh key to the vast.ai account. */
  'remote:addSshKey': () => RemoteState

  'remote:searchOffers': (filter: OfferFilter) => RemoteOffer[]
  /** Every instance of the account. */
  'remote:instances': () => VastInstance[]

  // The four that change billing. Window only.
  'remote:rent': (req: RentRequest) => RemoteState
  'remote:start': (rentalId: number) => RemoteState
  /** `force` stops it although it is in use. */
  'remote:stop': (rentalId: number, force: boolean) => RemoteState
  /** Rentals made by the app only. `label` must be the rental's label. */
  'remote:destroy': (rentalId: number, label: string, force: boolean) => RemoteState

  /** Looks again for an unsettled rent. */
  'remote:checkIntent': (label: string) => RemoteState
  /** Forgets an unresolved rent after the user looked on vast.ai's site. */
  'remote:dismissIntent': (label: string) => RemoteState

  /** Uses an instance rented elsewhere. */
  'remote:attach': (instanceId: number) => RemoteState
  /** Stops using an attached instance (it keeps running), or forgets one that is gone. */
  'remote:detach': (rentalId: number) => RemoteState
  'remote:setAutoStop': (rentalId: number, patch: Partial<AutoStop>) => RemoteState
  /** Cancels a countdown and counts as use now. */
  'remote:keepRunning': (rentalId: number) => RemoteState
  'remote:extendCap': (rentalId: number, hours: number) => RemoteState

  /** Installs the engine and, with a model, its files. Progress arrives as remote:changed. */
  'remote:setup': (rentalId: number, modelId: string | null) => RemoteState
  'remote:cancelSetup': (rentalId: number) => RemoteState
  /** Deletes a model's files on the instance. */
  'remote:removeModelFiles': (rentalId: number, modelId: string) => RemoteState
  /** The rental's setup and connection log, newest last. */
  'remote:log': (rentalId: number) => string[]

  'remote:saveModel': (def: Omit<RemoteModelDef, 'id' | 'createdAt'> & { id?: string }) => RemoteState
  'remote:deleteModel': (modelId: string) => RemoteState
  /** Loads a set-up model on a running rental. */
  'remote:load': (rentalId: number, modelId: string, config: LoadConfig | null) => LoadedInstance
  'remote:estimate': (rentalId: number, modelId: string, config: LoadConfig) => MemoryEstimate | null
}

export interface RemoteIpcEvents {
  'remote:changed': RemoteState
}

/** The channels that start, stop or end billing (docs/vast-design.md, section 2). */
export const BILLING_CHANNELS = ['remote:rent', 'remote:start', 'remote:stop', 'remote:destroy'] as const
