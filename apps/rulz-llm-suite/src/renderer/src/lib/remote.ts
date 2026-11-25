// Pure helpers for the vast.ai window: money and time words, button states, target checks, next-step sentences.
// Nothing here talks to the main process or reads a store, so all of it is tested (tests/renderer-remote.test.ts).
import {
  DEFAULT_AUTO_STOP,
  DEFAULT_BATCH_TARGET,
  DEFAULT_OFFER_FILTER,
  MAX_LANES,
  MAX_SESSIONS_PER_HOST,
  OFFER_SORTS,
  isRemoteModelId,
  type AutoStop,
  type BatchHost,
  type BatchTarget,
  type OfferFilter,
  type OfferSort,
  type RemoteModelDef,
  type RemoteModelSource,
  type RemoteOffer,
  type RemoteState,
  type RentIntent,
  type Rental
} from '@shared/remote'
import { MAX_CLOUD_SESSIONS, isCloudModelId } from '@shared/cloud'
import { RE_MIN_CONTEXT } from '@shared/re'
import type { LoadConfig, ModelEntry } from '@shared/types'

const GIB = 1024 ** 3

// ---------- Money and time ----------

/** "$0.42", "$12.30"; under a cent "<$0.01"; null = "—". */
export function money(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '—'
  if (n > 0 && n < 0.005) return '<$0.01'
  return `$${n.toFixed(2)}`
}

/** "$0.42/hour" for sentences, "$0.42/h" for tight places. */
export function perHour(n: number | null | undefined, short = false): string {
  return `${money(n)}/${short ? 'h' : 'hour'}`
}

/** "9 h 12 min", "45 min", "30 s", "2 d 3 h". Negative or null = "—". */
export function formatSpan(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms) || ms < 0) return '—'
  const s = Math.floor(ms / 1000)
  if (s < 60) return `${s} s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m} min`
  const h = Math.floor(m / 60)
  if (h < 24) return m % 60 ? `${h} h ${m % 60} min` : `${h} h`
  const d = Math.floor(h / 24)
  return h % 24 ? `${d} d ${h % 24} h` : `${d} d`
}

/** Hours the balance lasts at a price; null when either is unknown or the price is zero. */
export function balanceHours(credit: number | null, dollarsPerHour: number): number | null {
  if (credit === null || !Number.isFinite(credit) || !(dollarsPerHour > 0)) return null
  return Math.max(0, credit) / dollarsPerHour
}

export function balanceSentence(credit: number | null, dollarsPerHour: number): string {
  const h = balanceHours(credit, dollarsPerHour)
  if (h === null) return 'The account balance could not be read.'
  return `Balance ${money(credit)}: lasts about ${formatSpan(h * 3_600_000)} at this price.`
}

// ---------- Rental words ----------

export type Tone = 'neutral' | 'accent' | 'warn' | 'danger' | 'ok'

/** "RTX 4090 (12345)", "2× RTX 4090 (12345)": the label a batch item and a model's host badge show. */
export function rentalLabel(r: Pick<Rental, 'id' | 'gpuName' | 'numGpus'>): string {
  return `${r.numGpus > 1 ? `${r.numGpus}× ` : ''}${r.gpuName} (${r.id})`
}

export function gpuSummary(r: Pick<Rental, 'gpuName' | 'numGpus' | 'gpuRamMiB'>): string {
  return `${r.numGpus > 1 ? `${r.numGpus}× ` : ''}${r.gpuName}, ${Math.round((r.gpuRamMiB * r.numGpus) / 1024)} GB`
}

export type StatusIcon = 'running' | 'starting' | 'stopping' | 'stopped' | 'gone' | 'unknown'

export function statusWords(r: Pick<Rental, 'rentalStatus' | 'statusMessage'>): { label: string; tone: Tone; icon: StatusIcon } {
  switch (r.rentalStatus) {
    case 'running':
      return { label: 'Running', tone: 'ok', icon: 'running' }
    case 'starting':
      return { label: 'Starting', tone: 'accent', icon: 'starting' }
    case 'stopping':
      return { label: 'Stopping', tone: 'warn', icon: 'stopping' }
    case 'stopped':
      return { label: 'Stopped (the disk is still billed)', tone: 'neutral', icon: 'stopped' }
    case 'gone':
      return { label: 'No longer on vast.ai', tone: 'danger', icon: 'gone' }
    default:
      return { label: 'Status unknown', tone: 'warn', icon: 'unknown' }
  }
}

export function sshWords(r: Pick<Rental, 'ssh' | 'rentalStatus'>): { label: string; tone: Tone } {
  if (r.rentalStatus !== 'running') return { label: 'ssh: not connected (not running)', tone: 'neutral' }
  switch (r.ssh.state) {
    case 'ok':
      return { label: 'ssh: connected', tone: 'ok' }
    case 'connecting':
      return { label: 'ssh: connecting', tone: 'accent' }
    case 'failed':
      return { label: `ssh: failed${r.ssh.error ? `: ${r.ssh.error}` : ''}`, tone: 'danger' }
    default:
      return { label: 'ssh: not checked yet', tone: 'neutral' }
  }
}

export function engineWords(r: Pick<Rental, 'engine'>): { label: string; tone: Tone } {
  const e = r.engine
  switch (e.state) {
    case 'ready':
      return { label: `Engine ${e.tag || ''} ready`.replace('  ', ' '), tone: 'ok' }
    case 'working':
      return { label: `Engine: ${e.stage || 'installing'}`, tone: 'accent' }
    case 'failed':
      return { label: `Engine failed${e.error ? `: ${e.error}` : ''}`, tone: 'danger' }
    default:
      return { label: 'Engine not installed', tone: 'neutral' }
  }
}

export function modelStateWords(state: 'none' | 'working' | 'ready' | 'failed'): { label: string; tone: Tone } {
  switch (state) {
    case 'ready':
      return { label: 'Set up', tone: 'ok' }
    case 'working':
      return { label: 'Setting up', tone: 'accent' }
    case 'failed':
      return { label: 'Set-up failed', tone: 'danger' }
    default:
      return { label: 'Not set up', tone: 'neutral' }
  }
}

/** Is a rental ready for a model to be loaded on it: running, connected, engine installed. */
export function canLoadOn(r: Rental, modelId: string): boolean {
  return r.rentalStatus === 'running' && r.ssh.state === 'ok' && r.engine.state === 'ready' && r.models.some((m) => m.modelId === modelId && m.state === 'ready')
}

// ---------- Auto-stop line and countdown ----------

export interface AutoStopLine {
  text: string
  tone: 'plain' | 'warn'
  /** The text is a warning badge ("Cannot stop itself: ..."), not a plain line. */
  badge: boolean
}

export function autoStopLine(r: Rental, now: number): AutoStopLine {
  const a = r.autoStop
  if (r.rentalStatus === 'stopped') return { text: 'Stopped: the disk is billed until you start or destroy it.', tone: 'plain', badge: false }
  if (!a.enabled && a.capHours === null) return { text: 'Auto-stop is off', tone: 'warn', badge: true }
  if (a.enabled && r.watchdog.state === 'not-armed') {
    return { text: `Cannot stop itself${r.watchdog.error ? `: ${r.watchdog.error}` : ''}. Only this app's timer protects it.`, tone: 'warn', badge: true }
  }
  const parts: string[] = []
  if (a.enabled) {
    if (r.inUse.length) parts.push(`Stays on while in use; stops itself ${a.idleMinutes} idle minutes after`)
    else parts.push(`Stops itself after ${a.idleMinutes} idle minutes`)
  }
  if (r.capAt !== null) parts.push(`${a.enabled ? 'at the latest' : 'Stops itself'} in ${formatSpan(r.capAt - now)}`)
  else if (a.capHours === null && a.enabled) parts.push('no time cap')
  if (!parts.length) return { text: 'Auto-stop is off', tone: 'warn', badge: true }
  const text = parts.length === 2 && a.enabled ? `${parts[0]}; ${parts[1]}` : parts.join('; ')
  return { text, tone: 'plain', badge: false }
}

/** Seconds left of the app's stop countdown; null = none. */
export function countdownSeconds(r: Pick<Rental, 'countdown'>, now: number): number | null {
  if (!r.countdown) return null
  return Math.max(0, Math.ceil((r.countdown.at - now) / 1000))
}

/** Settings the auto-stop dialog saved: positive whole numbers; a cleared cap is no cap. */
export function cleanAutoStop(a: AutoStop): AutoStop {
  const idle = Number.isFinite(a.idleMinutes) ? Math.min(24 * 60, Math.max(1, Math.round(a.idleMinutes))) : 30
  const cap = a.capHours === null || !Number.isFinite(a.capHours) || a.capHours <= 0 ? null : Math.min(24 * 30, Math.round(a.capHours * 10) / 10)
  return { enabled: a.enabled, idleMinutes: idle, capHours: cap, apiCountsAsUse: a.apiCountsAsUse }
}

// ---------- Totals and the top-bar chip ----------

export interface Totals {
  runningCount: number
  /** $/hour of everything running or starting. */
  perHour: number
  /** Estimated cost of the running periods so far. */
  spent: number
  stoppedCount: number
  /** What the stopped ones cost per day for their disk. */
  stoppedPerDay: number
}

const isBilling = (r: Rental): boolean => r.rentalStatus === 'running' || r.rentalStatus === 'starting' || r.rentalStatus === 'stopping'

export function totals(rentals: Rental[]): Totals {
  const t: Totals = { runningCount: 0, perHour: 0, spent: 0, stoppedCount: 0, stoppedPerDay: 0 }
  for (const r of rentals) {
    if (isBilling(r)) {
      t.runningCount++
      t.perHour += r.dollarsPerHour
      t.spent += r.spent
    } else if (r.rentalStatus === 'stopped') {
      t.stoppedCount++
      t.stoppedPerDay += (r.storageDollarsPerHour ?? 0) * 24
    }
  }
  return t
}

export interface Chip {
  kind: 'none' | 'running' | 'stopped'
  text: string
  /** One line per rental. */
  lines: string[]
  /** A rental the app is about to stop. */
  countdown: { rentalId: number; seconds: number; more: number } | null
}

export function chipFor(rentals: Rental[], now: number): Chip {
  const t = totals(rentals)
  const lines = rentals
    .filter((r) => r.rentalStatus !== 'gone')
    .map((r) => {
      if (isBilling(r)) return `${rentalLabel(r)}: ${perHour(r.dollarsPerHour)}, ${money(r.spent)} so far`
      if (r.rentalStatus === 'stopped') {
        return `${rentalLabel(r)}: stopped, disk ${r.storageDollarsPerHour === null ? 'price not reported' : `${perHour(r.storageDollarsPerHour)} (${money(r.storageDollarsPerHour * 24)}/day)`}`
      }
      return `${rentalLabel(r)}: status unknown`
    })
  // The one that stops first; after Keep running the next one shows.
  const due = rentals.filter((r) => r.countdown).sort((a, b) => a.countdown!.at - b.countdown!.at)
  const countdown = due.length ? { rentalId: due[0].id, seconds: countdownSeconds(due[0], now) ?? 0, more: due.length - 1 } : null
  if (t.runningCount) return { kind: 'running', text: `${perHour(t.perHour, true)} · ${money(t.spent)}`, lines, countdown }
  if (t.stoppedCount) {
    const text = t.stoppedPerDay > 0 ? `${t.stoppedCount} stopped · ${money(t.stoppedPerDay)}/day for disks` : `${t.stoppedCount} stopped`
    return { kind: 'stopped', text, lines, countdown }
  }
  return { kind: 'none', text: '', lines: [], countdown: null }
}

// ---------- Rent: filter, offers, confirm ----------

/** Minimum GB of VRAM a model of this file size needs: the weights plus about 15% and 3 GB for the context. */
export function minVramForModel(sizeBytes: number | null | undefined): number | null {
  if (!sizeBytes || !(sizeBytes > 0)) return null
  return Math.ceil((sizeBytes / GIB) * 1.15 + 3)
}

/** The size of a remote model's files when known: its library file for a local source, else the entry the model list gives. */
export function remoteModelSize(def: RemoteModelDef, models: ModelEntry[]): number | null {
  if (def.source.kind === 'local') {
    const src = def.source.modelId
    const m = models.find((x) => x.id === src && !x.remote)
    if (m) return m.sizeBytes + (m.mmproj?.sizeBytes ?? 0)
  }
  const e = models.find((x) => x.id === def.id)
  return e && e.sizeBytes > 0 ? e.sizeBytes : null
}

/** vast.ai's DLPerf/$/hr; null when the speed score or a price is missing. */
export function dlperfPerDollar(o: Pick<RemoteOffer, 'dlperf' | 'dollarsPerHour'>): number | null {
  return o.dlperf === null || !(o.dollarsPerHour > 0) ? null : o.dlperf / o.dollarsPerHour
}

/** Offers without the value come last in both directions. */
export function sortOffers(offers: RemoteOffer[], key: OfferSort, dir: SortDir): RemoteOffer[] {
  const val = (o: RemoteOffer): number | null =>
    key === 'price' ? o.dollarsPerHour : key === 'value' ? dlperfPerDollar(o) : key === 'vram' ? o.gpuRamMiB * o.numGpus : key === 'reliability' ? o.reliability : o.inetDownMbps
  const sign = dir === 'asc' ? 1 : -1
  return [...offers].sort((a, b) => {
    const x = val(a)
    const y = val(b)
    const main = x === null || y === null ? Number(x === null) - Number(y === null) : sign * (x - y)
    return main || a.dollarsPerHour - b.dollarsPerHour || a.id - b.id
  })
}

export type SortDir = 'asc' | 'desc'

// ---------- What the rent window remembers ----------

/** Auto-stop itself is not part of it: it is on again for every new rental. */
export interface RentPrefs {
  filter: OfferFilter
  dir: SortDir
  /** Remote model the search is for; '' = none. */
  forModel: string
  idleMinutes: number
  capHours: number
}

/** The last used settings from their saved text; anything missing or out of range becomes the default. */
export function readRentPrefs(text: string | null): RentPrefs {
  let raw: Record<string, unknown> = {}
  try {
    const p: unknown = JSON.parse(text ?? '')
    if (p && typeof p === 'object') raw = p as Record<string, unknown>
  } catch {
    // Nothing saved yet, or not readable: the defaults.
  }
  const f = (raw.filter && typeof raw.filter === 'object' ? raw.filter : {}) as Record<string, unknown>
  const d = DEFAULT_OFFER_FILTER
  const n = (v: unknown, def: number, lo: number, hi: number): number => (typeof v === 'number' && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : def)
  return {
    filter: {
      minVramGb: n(f.minVramGb, d.minVramGb, 1, 2000),
      gpuName: typeof f.gpuName === 'string' ? f.gpuName.slice(0, 60) : d.gpuName,
      maxGpus: Math.round(n(f.maxGpus, d.maxGpus, 1, 8)),
      maxDollarsPerHour: f.maxDollarsPerHour === null ? null : n(f.maxDollarsPerHour, d.maxDollarsPerHour ?? 1, 0, 100),
      diskGb: Math.round(n(f.diskGb, d.diskGb, 10, 4000)),
      minReliability: n(f.minReliability, d.minReliability, 0, 1),
      minInetDownMbps: n(f.minInetDownMbps, d.minInetDownMbps, 0, 100_000),
      sort: OFFER_SORTS.find((s) => s === f.sort) ?? d.sort
    },
    dir: raw.dir === 'desc' ? 'desc' : 'asc',
    forModel: typeof raw.forModel === 'string' ? raw.forModel : '',
    idleMinutes: Math.round(n(raw.idleMinutes, DEFAULT_AUTO_STOP.idleMinutes, 1, 1440)),
    capHours: n(raw.capHours, DEFAULT_AUTO_STOP.capHours ?? 12, 1, 720)
  }
}

export function offerVramGb(o: Pick<RemoteOffer, 'gpuRamMiB' | 'numGpus'>): number {
  return Math.round((o.gpuRamMiB * o.numGpus) / 1024)
}

export const rentButtonLabel = (dollarsPerHour: number): string => `Rent for ${perHour(dollarsPerHour)}`

/** What the download costs at the offer's traffic price; null when the size or the price is not known. */
export function downloadCost(sizeBytes: number | null, inetDownPerGb: number | null): number | null {
  if (!sizeBytes || inetDownPerGb === null) return null
  return (sizeBytes / GIB) * inetDownPerGb
}

export interface RentSummary {
  /** One sentence per fact the dialog states; every one is shown. */
  lines: string[]
  warnings: string[]
}

/** The facts the rent confirmation must state before the user pays. */
export function rentSummary(o: RemoteOffer, opts: { diskGb: number; autoStop: AutoStop; credit: number | null; modelBytes: number | null }): RentSummary {
  const lines = [
    `${o.numGpus > 1 ? `${o.numGpus}× ` : ''}${o.gpuName}, ${offerVramGb(o)} GB of video memory, at ${perHour(o.dollarsPerHour)} (about ${money(o.dollarsPerHour * 24)} a day).`,
    `Disk: ${opts.diskGb} GB. A stopped instance still bills its disk until it is destroyed.`,
    `Traffic: ${o.inetDownPerGb === null ? 'download price not reported' : `${money(o.inetDownPerGb)} per GB received`}, ${o.inetUpPerGb === null ? 'upload price not reported' : `${money(o.inetUpPerGb)} per GB sent`}.`
  ]
  const dl = downloadCost(opts.modelBytes, o.inetDownPerGb)
  if (dl !== null) lines.push(`Downloading the model files costs about ${money(dl)} of traffic.`)
  const a = opts.autoStop
  lines.push(
    a.enabled
      ? `It stops itself after ${a.idleMinutes} idle minutes${a.capHours !== null ? ` and at the latest ${a.capHours} hours after it starts` : ''}.`
      : `Auto-stop is off${a.capHours !== null ? `; it still stops ${a.capHours} hours after it starts` : ''}.`
  )
  lines.push(balanceSentence(opts.credit, o.dollarsPerHour))
  const warnings: string[] = []
  if (!a.enabled && a.capHours === null) warnings.push('With auto-stop off and no time cap, it bills until you stop it yourself.')
  else if (!a.enabled) warnings.push('With auto-stop off, an idle instance bills until the time cap.')
  const h = balanceHours(opts.credit, o.dollarsPerHour)
  if (h !== null && h < 2) warnings.push('The balance lasts under 2 hours at this price.')
  return { lines, warnings }
}

/** Why renting is blocked right now; null = allowed. */
export function rentBlockedReason(s: Pick<RemoteState, 'intents' | 'vastKeySource'> | null): string | null {
  if (!s) return 'The vast.ai state is not loaded yet.'
  if (!s.vastKeySource) return 'Add a vast.ai API key first.'
  if (s.intents.length) return 'Renting is blocked until the earlier rent request is settled (see below).'
  return null
}

export function intentSentence(i: RentIntent, now: number): string {
  const age = formatSpan(now - i.createdAt)
  if (i.state === 'sending') return `Sending the rent request for ${perHour(i.dollarsPerHour)} (${age} ago).`
  if (i.state === 'checking') return `The answer was lost, so the app is checking whether it was rented (${age} so far). It may have been rented at ${perHour(i.dollarsPerHour)}.`
  return `It may have been rented at ${perHour(i.dollarsPerHour)}: the app could not find out after ${age}. Look at your instances on vast.ai before you forget this.`
}

// ---------- Which buttons a rental has ----------

export interface RentalActions {
  start: boolean
  stop: boolean
  destroy: boolean
  stopUsing: boolean
  forget: boolean
  setupEngine: boolean
  keepRunning: boolean
  extendCap: boolean
  autoStopSettings: boolean
}

export function rentalActions(r: Rental): RentalActions {
  const running = r.rentalStatus === 'running' || r.rentalStatus === 'starting'
  const gone = r.rentalStatus === 'gone'
  const busy = r.rentalStatus === 'stopping'
  return {
    start: r.rentalStatus === 'stopped',
    stop: running,
    destroy: r.origin === 'app' && !gone && !busy,
    stopUsing: r.origin === 'attached' && !gone,
    forget: gone,
    setupEngine: r.rentalStatus === 'running' && r.ssh.state === 'ok' && r.engine.state !== 'ready' && r.engine.state !== 'working',
    keepRunning: r.countdown !== null,
    extendCap: !gone && r.capAt !== null,
    autoStopSettings: !gone
  }
}

export interface ConfirmText {
  title: string
  lines: string[]
  button: string
  /** The rental is in use: after this the user is asked again, and only that answer sends force. */
  asksAgain: boolean
}

export function stopConfirmText(r: Rental): ConfirmText {
  const lines = [
    `This stops ${rentalLabel(r)}. Billing for the GPU stops; the disk (${r.diskGb} GB${r.storageDollarsPerHour === null ? '' : `, ${perHour(r.storageDollarsPerHour)}`}) stays and is billed until you destroy it.`,
    'Models loaded on it are unloaded. Their files stay on the disk, so starting it again is quick.'
  ]
  if (r.foreignProcesses) lines.push('Something else is using its GPU. That stops too.')
  if (r.origin === 'attached') lines.push('It was rented elsewhere. Anything else running on it stops with it.')
  return { title: 'Stop this instance?', lines, button: 'Stop', asksAgain: r.inUse.length > 0 }
}

export function destroyConfirmText(r: Rental): ConfirmText {
  const lines = [
    `This destroys "${r.label}" (${rentalLabel(r)}) on vast.ai. Billing ends completely.`,
    `Everything on its disk is lost: ${r.diskGb} GB, including the engine and ${r.models.length ? `the files of ${r.models.length} model${r.models.length === 1 ? '' : 's'}` : 'any model files'}. This cannot be undone.`
  ]
  return { title: 'Destroy this instance?', lines, button: 'Destroy', asksAgain: r.inUse.length > 0 }
}

/** The second question for a rental that is in use: what is using it, and that the use is cancelled cleanly. */
export function inUseText(r: Rental, verb: 'stop' | 'destroy'): ConfirmText {
  return {
    title: `It is in use now. ${verb === 'stop' ? 'Stop' : 'Destroy'} it anyway?`,
    lines: [
      `${rentalLabel(r)} is in use now: ${r.inUse.join(', ')}.`,
      `${verb === 'stop' ? 'Stopping' : 'Destroying'} it cancels that work cleanly: batch tasks go back to not run, and models are unloaded first.`
    ],
    button: verb === 'stop' ? 'Stop it anyway' : 'Destroy it anyway',
    asksAgain: false
  }
}

// ---------- Batch target ----------

export interface HostChoice {
  host: BatchHost
  label: string
  /** The model can run here and the host is ready. */
  valid: boolean
  /** Why not, in a sentence; '' when valid. */
  why: string
}

/** Does a model id name a remote model (the run then uses rentals only); a local one runs on this PC only. */
export function hostKindFor(modelId: string | null | undefined): 'local' | 'vast' | 'cloud' {
  if (modelId && isCloudModelId(modelId)) return 'cloud'
  return isRemoteModelId(modelId) ? 'vast' : 'local'
}

/** The model a batch run uses: the one in the RE settings if the list has it, else the active one (the batch store's rule). */
export function batchModelId(wanted: string | null | undefined, models: { id: string }[], activeModelId: string | null): string | null {
  return wanted && models.some((m) => m.id === wanted) ? wanted : activeModelId
}

/**
 * Every host a run could use, with why a host is not usable. A model at OpenRouter (`cloud`) has exactly one choice,
 * OpenRouter, and runs nowhere else; with a REA that is not stdio it cannot run (the rental rule: private REA per task).
 */
export function hostChoices(modelId: string | null, rentals: Rental[], reIsStdio: boolean, cloud = false): HostChoice[] {
  if (cloud) {
    const why = reIsStdio ? '' : 'REA must be a stdio server to run tasks on OpenRouter models.'
    return [{ host: { kind: 'cloud' }, label: 'OpenRouter', valid: !why, why }]
  }
  const remote = isRemoteModelId(modelId)
  const out: HostChoice[] = []
  out.push({
    host: { kind: 'local' },
    label: 'This PC',
    valid: !remote,
    why: remote ? 'A model on vast.ai runs on rentals only.' : ''
  })
  for (const r of rentals) {
    if (r.rentalStatus === 'gone') continue
    let why = ''
    if (!remote) why = 'A model from this PC runs on this PC only.'
    else if (r.rentalStatus !== 'running') why = `${rentalLabel(r)} is not running. Start it on the vast.ai page first.`
    else if (!modelId || !r.models.some((m) => m.modelId === modelId && m.state === 'ready')) why = `${rentalLabel(r)} has no set-up copy of this model. Set it up on the vast.ai page.`
    else if (!reIsStdio) why = 'REA must be a stdio server to run several tasks at once.'
    out.push({ host: { kind: 'vast', rentalId: r.id }, label: rentalLabel(r), valid: !why, why })
  }
  return out
}

export const hostKey = (h: BatchHost): string => (h.kind === 'local' ? 'local' : h.kind === 'cloud' ? 'cloud' : `vast:${h.rentalId}`)

/** Most tasks at once on each host for this many hosts; OpenRouter (`cloud`) allows up to MAX_CLOUD_SESSIONS. */
export function maxSessions(hostCount: number, cloud = false): number {
  const most = Math.max(1, Math.min(MAX_SESSIONS_PER_HOST, Math.floor(MAX_LANES / Math.max(1, hostCount))))
  return cloud ? Math.min(most, MAX_CLOUD_SESSIONS) : most
}

export const CONTEXT_PER_SESSION_PARALLEL = 65536
export const REA_GB_PER_TASK = 1.3

/**
 * The context a model is saved to load with for reverse engineering, to share out over a batch run's tasks: the mode's
 * own load settings', else the model's default ones' when those are saved (`custom`) and enough for one task: a small
 * context saved for chatting says nothing about practice. null = not known (also auto, or the app's defaults).
 */
export function knownReContext(
  reLoad: LoadConfig | null | undefined,
  base: { config: LoadConfig; custom: boolean } | null | undefined
): { total: number; from: 're' | 'default' } | null {
  const ok = (n: number | null | undefined): n is number => typeof n === 'number' && Number.isFinite(n) && n > 0
  if (reLoad && ok(reLoad.contextLength)) return { total: reLoad.contextLength, from: 're' }
  if (base?.custom && ok(base.config.contextLength) && base.config.contextLength >= RE_MIN_CONTEXT) return { total: base.config.contextLength, from: 'default' }
  return null
}

/** The most tasks at once (up to 4, within `most`) that each get at least RE_MIN_CONTEXT of `total`; at least 1. */
function tasksFitting(total: number, most: number): number {
  for (let k = Math.min(4, most); k > 1; k--) if (Math.floor(total / k) >= RE_MIN_CONTEXT) return k
  return 1
}

/**
 * A target for the chosen hosts, with sessions and context defaults. Without a known context (`total`): 1 task for this
 * PC alone, 4 when a rental is in it, each with 65536 tokens. With one, it is the total the model loads with over all
 * tasks at once (llama-server splits it over its slots), so the default shares it out and never goes over it.
 */
export function targetFor(hosts: BatchHost[], prev?: Partial<BatchTarget>, total?: number | null): BatchTarget {
  const hs = hosts.length ? hosts : DEFAULT_BATCH_TARGET.hosts
  const hasRental = hs.some((h) => h.kind === 'vast')
  // OpenRouter: one task unless the user asks for more (every task costs money), and the context is the model's own.
  const cloud = hs.some((h) => h.kind === 'cloud')
  const most = maxSessions(hs.length, cloud)
  const known = total && total > 0 ? total : null
  const fallbackN = hasRental ? (known ? tasksFitting(known, most) : 4) : 1
  const n = Math.min(most, Math.max(1, prev?.sessionsPerHost ?? fallbackN))
  if (cloud) return { hosts: hs, sessionsPerHost: n, contextPerSession: null, stopRentalsAfter: false }
  const needCtx = n > 1 || hasRental
  const share = known ? (n === 1 ? known : Math.max(1024, Math.floor(known / n / 1024) * 1024)) : CONTEXT_PER_SESSION_PARALLEL
  const ctx = prev?.contextPerSession ?? (needCtx ? share : null)
  return { hosts: hs, sessionsPerHost: n, contextPerSession: needCtx ? (ctx ?? share) : ctx, stopRentalsAfter: hasRental ? (prev?.stopRentalsAfter ?? false) : false }
}

/** A sentence when a task gets less context than reverse engineering works well with; null = fine. Not a blocking problem. */
export function contextWarning(t: Pick<BatchTarget, 'contextPerSession'>): string | null {
  const c = t.contextPerSession
  if (c === null || c >= RE_MIN_CONTEXT) return null
  return `${c.toLocaleString('en-US')} tokens per task is little for reverse engineering; ${RE_MIN_CONTEXT.toLocaleString('en-US')} or more works much better.`
}

/** Sentences why the target cannot start; empty = it can. */
export function targetProblems(t: BatchTarget, choices: HostChoice[]): string[] {
  const out: string[] = []
  if (!t.hosts.length) out.push('Choose where the run goes.')
  for (const h of t.hosts) {
    const c = choices.find((x) => hostKey(x.host) === hostKey(h.kind === 'local' ? { kind: 'local' } : h))
    if (!c) out.push('A chosen host is no longer there.')
    else if (!c.valid) out.push(c.why)
  }
  const cloud = t.hosts.some((h) => h.kind === 'cloud')
  if (t.sessionsPerHost > maxSessions(Math.max(1, t.hosts.length), cloud)) out.push(`At most ${maxSessions(t.hosts.length, cloud)} tasks at once on each.`)
  // A model at OpenRouter has its own context (it is not loaded), so tasks at once need no context share.
  if (!cloud && (t.sessionsPerHost > 1 || t.hosts.some((h) => h.kind === 'vast')) && t.contextPerSession === null) out.push('Set the context per session.')
  return out
}

/** Hosts of a saved target that are still valid choices; the rest dropped. Null when none are left (the user must choose). */
export function resumeTarget(old: BatchTarget | undefined, choices: HostChoice[]): BatchTarget | null {
  if (!old) return null
  const hosts = old.hosts.filter((h) => choices.some((c) => c.valid && hostKey(c.host) === hostKey(h)))
  if (hosts.length !== old.hosts.length || !hosts.length) return null
  return { ...old, hosts }
}

/** The RAM sentence for tasks at once on this PC. */
export function ramHint(sessionsOnThisPc: number): string {
  const gb = sessionsOnThisPc * REA_GB_PER_TASK
  return `About ${REA_GB_PER_TASK} GB of this PC's memory per task at once, for REA${sessionsOnThisPc > 1 ? `: about ${gb.toFixed(1)} GB for ${sessionsOnThisPc}` : ''}.`
}

/** $ per solved task of a finished run; null when nothing was solved or no cost is known. */
export function costPerSolved(cost: number | undefined, solved: number): number | null {
  if (cost === undefined || !(solved > 0)) return null
  return cost / solved
}

// ---------- Loading a remote model from the picker ----------

export type LoadPlan = { kind: 'load'; rentalId: number } | { kind: 'go'; sentence: string }

/**
 * What choosing a remote model that is not loaded does: load it where exactly one running rental has it set up,
 * otherwise send the user to the vast.ai page. It never rents or starts anything.
 */
export function loadPlan(modelId: string, rentals: Rental[]): LoadPlan {
  const ready = rentals.filter((r) => canLoadOn(r, modelId))
  if (ready.length === 1) return { kind: 'load', rentalId: ready[0].id }
  if (ready.length > 1) return { kind: 'go', sentence: 'This model is set up on more than one rental. Choose which one to load it on, on the vast.ai page.' }
  const running = rentals.filter((r) => r.rentalStatus === 'running')
  if (!rentals.length) return { kind: 'go', sentence: 'This model runs on a rented vast.ai instance, and none is rented yet. Rent one on the vast.ai page.' }
  if (!running.length) return { kind: 'go', sentence: 'No vast.ai instance is running. Start or rent one on the vast.ai page, then set this model up on it.' }
  return { kind: 'go', sentence: 'No running vast.ai instance has this model set up. Set it up on the vast.ai page.' }
}

/** Remote models that are loaded, shown with their host: the badge text. */
export const hostBadge = (host: { label: string } | undefined): string => (host ? host.label : 'This PC')

/** A sentence for the chat banner of a pool-owned chat. */
export const POOL_CHAT_TEXT = 'A batch run is using this chat'

// ---------- Remote model definitions ----------

/** Why a source cannot be saved, in a sentence; null = fine. */
export function validateRemoteSource(src: RemoteModelSource): string | null {
  if (src.kind === 'local') return src.modelId ? null : 'Choose a model from your library.'
  if (!/^[\w.-]+\/[\w.-]+$/.test(src.repo.trim())) return `Write the ${src.bucket ? 'bucket' : 'repo'} as owner/name, for example unsloth/Qwen3-GGUF.`
  if (!src.file.trim()) return 'Name the model file. For a split model, the first part.'
  if (/[\\\s]/.test(src.file.trim()) || src.file.includes('..')) return 'The file name has spaces, backslashes or "..", which Hugging Face paths do not.'
  if (src.mmproj.trim() && (/[\\\s]/.test(src.mmproj.trim()) || src.mmproj.includes('..'))) return 'The projector file name has spaces, backslashes or "..".'
  return null
}

/** A name for a new definition from its source. */
export function defaultModelName(src: RemoteModelSource, library: ModelEntry[]): string {
  if (src.kind === 'local') return library.find((m) => m.id === src.modelId)?.name ?? src.modelId.split('/').pop() ?? ''
  const file = src.file.trim().split('/').pop() ?? ''
  return file.replace(/\.gguf$/i, '').replace(/-0*1-of-\d+$/i, '') || src.repo.trim().split('/').pop() || ''
}

/** A source with its text fields trimmed; an empty revision means main. */
export function cleanRemoteSource(src: RemoteModelSource): RemoteModelSource {
  if (src.kind === 'local') return { kind: 'local', modelId: src.modelId }
  return { kind: 'hf', repo: src.repo.trim(), bucket: src.bucket, revision: src.revision.trim() || 'main', file: src.file.trim(), mmproj: src.mmproj.trim() }
}

export function sourceSummary(src: RemoteModelSource): string {
  if (src.kind === 'local') return `From this PC: ${src.modelId}`
  return `${src.bucket ? 'Bucket' : 'Hugging Face'} ${src.repo}${src.revision && src.revision !== 'main' ? ` @ ${src.revision}` : ''}: ${src.file}${src.mmproj ? ` + ${src.mmproj}` : ''}`
}
