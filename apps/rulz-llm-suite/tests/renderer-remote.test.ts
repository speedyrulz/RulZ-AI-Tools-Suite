import { describe, expect, it } from 'vitest'
import { DEFAULT_LOAD_CONFIG } from '../src/shared/defaults'
import type { LoadConfig } from '../src/shared/types'
import { DEFAULT_AUTO_STOP, DEFAULT_BATCH_TARGET, DEFAULT_OFFER_FILTER, type BatchHost, type RemoteModelDef, type RemoteOffer, type Rental } from '../src/shared/remote'
import {
  autoStopLine,
  balanceHours,
  balanceSentence,
  batchModelId,
  canLoadOn,
  chipFor,
  cleanAutoStop,
  cleanRemoteSource,
  costPerSolved,
  countdownSeconds,
  defaultModelName,
  destroyConfirmText,
  downloadCost,
  formatSpan,
  hostChoices,
  inUseText,
  intentSentence,
  loadPlan,
  maxSessions,
  minVramForModel,
  money,
  perHour,
  ramHint,
  remoteModelSize,
  rentBlockedReason,
  rentButtonLabel,
  rentalActions,
  rentalLabel,
  rentSummary,
  resumeTarget,
  dlperfPerDollar,
  readRentPrefs,
  sortOffers,
  statusWords,
  stopConfirmText,
  contextWarning,
  knownReContext,
  targetFor,
  targetProblems,
  totals,
  validateRemoteSource
} from '../src/renderer/src/lib/remote'

const GIB = 1024 ** 3
const NOW = 1_000_000_000_000

function rental(over: Partial<Rental> = {}): Rental {
  return {
    id: 101,
    label: 'rulz-llm-abcd1234',
    status: 'running',
    statusMessage: '',
    gpuName: 'RTX 4090',
    numGpus: 1,
    gpuRamMiB: 24576,
    dollarsPerHour: 0.42,
    storageDollarsPerHour: 0.01,
    diskGb: 60,
    startedAt: NOW - 3_600_000,
    image: 'x',
    origin: 'app',
    rentalStatus: 'running',
    ssh: { state: 'ok' },
    engine: { state: 'ready', tag: 'b12407', stage: '' },
    models: [{ modelId: 'vast/q', state: 'ready', bytesDone: 0, bytesTotal: 0, stage: '' }],
    gpus: [],
    diskFreeBytes: 40 * GIB,
    foreignProcesses: false,
    autoStop: { ...DEFAULT_AUTO_STOP },
    watchdog: { state: 'armed' },
    inUse: [],
    lastUsedAt: null,
    idleStopAt: null,
    capAt: NOW + (9 * 60 + 12) * 60_000,
    countdown: null,
    spent: 3.12,
    ...over
  }
}

function offer(over: Partial<RemoteOffer> = {}): RemoteOffer {
  return {
    id: 1,
    machineId: 1,
    gpuName: 'RTX 4090',
    numGpus: 1,
    gpuRamMiB: 24576,
    cpuCores: 16,
    cpuRamMiB: 65536,
    dollarsPerHour: 0.42,
    dlperf: 100,
    reliability: 0.99,
    inetDownMbps: 900,
    inetUpMbps: 500,
    inetDownPerGb: 0.01,
    inetUpPerGb: 0.02,
    cudaMaxGood: 12.8,
    geolocation: 'Sweden',
    maxDays: 30,
    ...over
  }
}

describe('money and time', () => {
  it('writes dollars with two decimals and a floor for tiny amounts', () => {
    expect(money(0.42)).toBe('$0.42')
    expect(money(12.3)).toBe('$12.30')
    expect(money(0.001)).toBe('<$0.01')
    expect(money(0)).toBe('$0.00')
    expect(money(null)).toBe('—')
    expect(perHour(0.42)).toBe('$0.42/hour')
    expect(perHour(0.42, true)).toBe('$0.42/h')
  })

  it('writes spans in words', () => {
    expect(formatSpan((9 * 60 + 12) * 60_000)).toBe('9 h 12 min')
    expect(formatSpan(45 * 60_000)).toBe('45 min')
    expect(formatSpan(30_000)).toBe('30 s')
    expect(formatSpan(2 * 3_600_000)).toBe('2 h')
    expect(formatSpan(51 * 3_600_000)).toBe('2 d 3 h')
    expect(formatSpan(-1)).toBe('—')
    expect(formatSpan(null)).toBe('—')
  })

  it('says how long the balance lasts', () => {
    expect(balanceHours(10, 0.5)).toBe(20)
    expect(balanceHours(null, 0.5)).toBeNull()
    expect(balanceHours(10, 0)).toBeNull()
    expect(balanceSentence(10, 0.5)).toContain('lasts about 20 h')
    expect(balanceSentence(null, 0.5)).toContain('could not be read')
  })
})

describe('rental words and buttons', () => {
  it('labels a rental with its GPU and id, and every status has a word', () => {
    expect(rentalLabel(rental())).toBe('RTX 4090 (101)')
    expect(rentalLabel(rental({ numGpus: 2 }))).toBe('2× RTX 4090 (101)')
    for (const s of ['starting', 'running', 'stopping', 'stopped', 'gone', 'unknown'] as const) {
      expect(statusWords({ rentalStatus: s, statusMessage: '' }).label.length).toBeGreaterThan(3)
    }
    expect(statusWords({ rentalStatus: 'stopped', statusMessage: '' }).label).toContain('disk')
  })

  it('offers Start only when stopped, Stop only when running, Destroy only for app rentals', () => {
    const run = rentalActions(rental())
    expect(run).toMatchObject({ start: false, stop: true, destroy: true, stopUsing: false })
    const stopped = rentalActions(rental({ rentalStatus: 'stopped' }))
    expect(stopped).toMatchObject({ start: true, stop: false, destroy: true })
    const attached = rentalActions(rental({ origin: 'attached' }))
    expect(attached).toMatchObject({ destroy: false, stopUsing: true })
    const gone = rentalActions(rental({ rentalStatus: 'gone' }))
    expect(gone).toMatchObject({ forget: true, start: false, stop: false, destroy: false })
    expect(rentalActions(rental({ countdown: { at: NOW + 1000, reason: 'idle' } })).keepRunning).toBe(true)
    expect(rentalActions(rental({ ssh: { state: 'ok' }, engine: { state: 'none', tag: '', stage: '' } })).setupEngine).toBe(true)
  })

  it('knows when a model can be loaded on a rental', () => {
    expect(canLoadOn(rental(), 'vast/q')).toBe(true)
    expect(canLoadOn(rental(), 'vast/other')).toBe(false)
    expect(canLoadOn(rental({ rentalStatus: 'stopped' }), 'vast/q')).toBe(false)
    expect(canLoadOn(rental({ ssh: { state: 'failed' } }), 'vast/q')).toBe(false)
    expect(canLoadOn(rental({ engine: { state: 'working', tag: '', stage: '' } }), 'vast/q')).toBe(false)
  })
})

describe('auto-stop line and countdown', () => {
  it('states the idle limit and the time left to the cap', () => {
    expect(autoStopLine(rental(), NOW)).toEqual({ text: 'Stops itself after 30 idle minutes; at the latest in 9 h 12 min', tone: 'plain', badge: false })
  })

  it('warns when it cannot stop itself or auto-stop is off', () => {
    const cannot = autoStopLine(rental({ watchdog: { state: 'not-armed', error: 'no API key on the instance' } }), NOW)
    expect(cannot.badge).toBe(true)
    expect(cannot.text).toContain('Cannot stop itself: no API key on the instance')
    const off = autoStopLine(rental({ autoStop: { ...DEFAULT_AUTO_STOP, enabled: false, capHours: null }, capAt: null }), NOW)
    expect(off).toMatchObject({ text: 'Auto-stop is off', badge: true, tone: 'warn' })
  })

  it('counts the stop countdown down to zero', () => {
    expect(countdownSeconds({ countdown: null }, NOW)).toBeNull()
    expect(countdownSeconds({ countdown: { at: NOW + 42_300, reason: '' } }, NOW)).toBe(43)
    expect(countdownSeconds({ countdown: { at: NOW - 5, reason: '' } }, NOW)).toBe(0)
  })

  it('cleans what the settings dialog saved', () => {
    expect(cleanAutoStop({ enabled: true, idleMinutes: 0, capHours: 0, apiCountsAsUse: false })).toMatchObject({ idleMinutes: 1, capHours: null })
    expect(cleanAutoStop({ enabled: false, idleMinutes: 12.6, capHours: 3.25, apiCountsAsUse: true })).toEqual({ enabled: false, idleMinutes: 13, capHours: 3.3, apiCountsAsUse: true })
    expect(cleanAutoStop({ enabled: true, idleMinutes: NaN, capHours: null, apiCountsAsUse: false }).idleMinutes).toBe(30)
  })
})

describe('totals and the top-bar chip', () => {
  it('adds up running rentals per hour and so far, and stopped ones per day', () => {
    const rs = [rental(), rental({ id: 102, dollarsPerHour: 0.84, spent: 1 }), rental({ id: 103, rentalStatus: 'stopped', storageDollarsPerHour: 0.02 })]
    const t = totals(rs)
    expect(t.runningCount).toBe(2)
    expect(t.perHour).toBeCloseTo(1.26)
    expect(t.spent).toBeCloseTo(4.12)
    expect(t.stoppedCount).toBe(1)
    expect(t.stoppedPerDay).toBeCloseTo(0.48)
  })

  it('writes the chip for running, stopped-only and no rentals', () => {
    const run = chipFor([rental({ dollarsPerHour: 0.84, spent: 3.12 })], NOW)
    expect(run.kind).toBe('running')
    expect(run.text).toBe('$0.84/h · $3.12')
    expect(run.lines[0]).toContain('RTX 4090 (101)')
    const quiet = chipFor([rental({ rentalStatus: 'stopped', storageDollarsPerHour: 0.02 })], NOW)
    expect(quiet.kind).toBe('stopped')
    expect(quiet.text).toBe('1 stopped · $0.48/day for disks')
    expect(quiet.lines[0]).toContain('stopped, disk')
    expect(chipFor([], NOW).kind).toBe('none')
    expect(chipFor([rental({ rentalStatus: 'gone' })], NOW).kind).toBe('none')
  })

  it('carries the countdown of a rental about to stop', () => {
    const c = chipFor([rental({ countdown: { at: NOW + 30_000, reason: 'idle' } })], NOW)
    expect(c.countdown).toEqual({ rentalId: 101, seconds: 30, more: 0 })
    const two = chipFor([rental({ id: 1, countdown: { at: NOW + 50_000, reason: 'idle' } }), rental({ id: 2, countdown: { at: NOW + 20_000, reason: 'cap' } })], NOW)
    expect(two.countdown).toEqual({ rentalId: 2, seconds: 20, more: 1 })
  })
})

describe('renting', () => {
  it('raises the minimum video memory from the model size', () => {
    expect(minVramForModel(20 * GIB)).toBe(26)
    expect(minVramForModel(0)).toBeNull()
    expect(minVramForModel(null)).toBeNull()
    expect(DEFAULT_OFFER_FILTER.minVramGb).toBe(24)
  })

  it('finds the size of a remote model from the library or its entry', () => {
    const lib = [
      { id: 'a/b.gguf', sizeBytes: 10 * GIB, mmproj: { sizeBytes: GIB }, remote: undefined },
      { id: 'vast/q', sizeBytes: 5 * GIB, remote: true }
    ] as never
    const local: RemoteModelDef = { id: 'vast/x', name: 'x', createdAt: 0, source: { kind: 'local', modelId: 'a/b.gguf' } }
    expect(remoteModelSize(local, lib)).toBe(11 * GIB)
    const hf: RemoteModelDef = { id: 'vast/q', name: 'q', createdAt: 0, source: { kind: 'hf', repo: 'o/n', bucket: false, revision: 'main', file: 'f.gguf', mmproj: '' } }
    expect(remoteModelSize(hf, lib)).toBe(5 * GIB)
    expect(remoteModelSize({ ...hf, id: 'vast/none' }, lib)).toBeNull()
  })

  it('sorts offers by price first and flips the direction', () => {
    const o = [offer({ id: 1, dollarsPerHour: 0.5 }), offer({ id: 2, dollarsPerHour: 0.3 }), offer({ id: 3, dollarsPerHour: 0.4 })]
    expect(sortOffers(o, 'price', 'asc').map((x) => x.id)).toEqual([2, 3, 1])
    expect(sortOffers(o, 'price', 'desc').map((x) => x.id)).toEqual([1, 3, 2])
    expect(sortOffers([offer({ id: 1, inetDownMbps: 100 }), offer({ id: 2, inetDownMbps: 900 })], 'down', 'desc')[0].id).toBe(2)
  })

  it('sorts by DLPerf per dollar, with offers that have no score last in both directions', () => {
    const o = [offer({ id: 1, dollarsPerHour: 0.5, dlperf: 100 }), offer({ id: 2, dlperf: null }), offer({ id: 3, dollarsPerHour: 1, dlperf: 400 }), offer({ id: 4, dollarsPerHour: 0, dlperf: 9 })]
    expect(dlperfPerDollar(o[0])).toBe(200)
    expect(dlperfPerDollar(o[3])).toBeNull()
    expect(sortOffers(o, 'value', 'desc').map((x) => x.id)).toEqual([3, 1, 4, 2])
    expect(sortOffers(o, 'value', 'asc').map((x) => x.id)).toEqual([1, 3, 4, 2])
  })

  it('reads the rent settings back, and falls back to the defaults for what is missing or wrong', () => {
    const none = readRentPrefs(null)
    expect(none).toEqual({ filter: DEFAULT_OFFER_FILTER, dir: 'asc', forModel: '', idleMinutes: 30, capHours: 12 })
    for (const bad of ['', '{', '3', 'null', '[]', '{"filter":5}']) expect(readRentPrefs(bad), bad).toEqual(none)
    const used = { filter: { ...DEFAULT_OFFER_FILTER, minVramGb: 48, gpuName: 'RTX 5090', maxDollarsPerHour: null, sort: 'value' as const }, dir: 'desc' as const, forModel: 'vast/q', idleMinutes: 10, capHours: 4 }
    expect(readRentPrefs(JSON.stringify(used))).toEqual(used)
    const odd = readRentPrefs(JSON.stringify({ filter: { maxGpus: 99, diskGb: 1, sort: 'x', minReliability: 'a' }, idleMinutes: -1, capHours: null }))
    expect(odd.filter).toEqual({ ...DEFAULT_OFFER_FILTER, maxGpus: 8, diskGb: 10 })
    expect(odd).toMatchObject({ idleMinutes: 1, capHours: 12 })
  })

  it('names the price on the rent button', () => {
    expect(rentButtonLabel(0.42)).toBe('Rent for $0.42/hour')
  })

  it('prices the download from the traffic price when both are known', () => {
    expect(downloadCost(10 * GIB, 0.02)).toBeCloseTo(0.2)
    expect(downloadCost(null, 0.02)).toBeNull()
    expect(downloadCost(10 * GIB, null)).toBeNull()
  })

  it('states every fact the confirmation needs', () => {
    const sum = rentSummary(offer(), { diskGb: 60, autoStop: { ...DEFAULT_AUTO_STOP }, credit: 8.4, modelBytes: 10 * GIB })
    const text = sum.lines.join('\n')
    expect(text).toContain('RTX 4090')
    expect(text).toContain('$0.42/hour')
    expect(text).toContain('$10.08 a day')
    expect(text).toContain('60 GB')
    expect(text).toContain('still bills its disk')
    expect(text).toContain('$0.01 per GB received')
    expect(text).toContain('Downloading the model files costs about $0.10')
    expect(text).toContain('30 idle minutes')
    expect(text).toContain('12 hours')
    expect(text).toContain('Balance $8.40: lasts about 20 h')
    expect(sum.warnings).toEqual([])
  })

  it('warns about no auto-stop and a short balance', () => {
    const sum = rentSummary(offer({ inetDownPerGb: null }), { diskGb: 60, autoStop: { ...DEFAULT_AUTO_STOP, enabled: false, capHours: null }, credit: 0.5, modelBytes: 10 * GIB })
    expect(sum.lines.join('\n')).toContain('download price not reported')
    expect(sum.lines.join('\n')).not.toContain('Downloading the model files')
    expect(sum.warnings.join('\n')).toContain('bills until you stop it yourself')
    expect(sum.warnings.join('\n')).toContain('under 2 hours')
  })

  it('blocks renting without a key and while a rent request is unsettled', () => {
    expect(rentBlockedReason(null)).toContain('not loaded')
    expect(rentBlockedReason({ intents: [], vastKeySource: '' })).toContain('API key')
    const intent = { label: 'rulz-llm-1', offerId: 1, dollarsPerHour: 0.4, createdAt: NOW, state: 'unresolved' as const, message: '' }
    expect(rentBlockedReason({ intents: [intent], vastKeySource: 'saved' })).toContain('blocked')
    expect(rentBlockedReason({ intents: [], vastKeySource: 'saved' })).toBeNull()
    expect(intentSentence(intent, NOW + 5 * 60_000)).toContain('may have been rented')
    expect(intentSentence({ ...intent, state: 'sending' }, NOW)).toContain('Sending')
  })
})

describe('stop and destroy questions', () => {
  it('says what stopping keeps and loses, and asks again only when it is in use', () => {
    const idle = stopConfirmText(rental())
    expect(idle.lines.join(' ')).toContain('disk')
    expect(idle.asksAgain).toBe(false)
    const busy = rental({ inUse: ['batch run'] })
    expect(stopConfirmText(busy).asksAgain).toBe(true)
    expect(inUseText(busy, 'stop').lines.join(' ')).toContain('batch run')
    expect(inUseText(busy, 'destroy').button).toBe('Destroy it anyway')
    expect(stopConfirmText(rental({ origin: 'attached', foreignProcesses: true })).lines.join(' ')).toContain('rented elsewhere')
  })

  it('names the label and the disk loss when destroying', () => {
    const d = destroyConfirmText(rental())
    expect(d.lines.join(' ')).toContain('rulz-llm-abcd1234')
    expect(d.lines.join(' ')).toContain('60 GB')
    expect(d.lines.join(' ')).toContain('cannot be undone')
  })
})

describe('batch target', () => {
  const models = (id: string): Rental[] => [rental({ id: 1 }), rental({ id: 2, rentalStatus: 'stopped' }), rental({ id: 3, models: [] })].map((r) => ({ ...r, models: r.models.map((m) => ({ ...m, modelId: id })) }))

  it('uses rentals only for a remote model and this PC only for a local one', () => {
    const remote = hostChoices('vast/q', models('vast/q'), true)
    expect(remote.find((c) => c.host.kind === 'local')).toMatchObject({ valid: false })
    expect(remote.filter((c) => c.valid).map((c) => c.label)).toEqual(['RTX 4090 (1)'])
    expect(remote.find((c) => c.label.includes('(2)'))?.why).toContain('not running')
    expect(remote.find((c) => c.label.includes('(3)'))?.why).toContain('no set-up copy')
    const local = hostChoices('Qwen/q.gguf', models('vast/q'), true)
    expect(local.find((c) => c.host.kind === 'local')).toMatchObject({ valid: true })
    expect(local.filter((c) => c.host.kind === 'vast').every((c) => !c.valid)).toBe(true)
  })

  it('refuses rentals when REA is not a stdio server', () => {
    const c = hostChoices('vast/q', models('vast/q'), false)
    expect(c.find((x) => x.label.includes('(1)'))).toMatchObject({ valid: false })
    expect(c.find((x) => x.label.includes('(1)'))?.why).toContain('stdio')
  })

  it('chooses the run model like the batch store does', () => {
    expect(batchModelId('a', [{ id: 'a' }], 'b')).toBe('a')
    expect(batchModelId('gone', [{ id: 'a' }], 'b')).toBe('b')
    expect(batchModelId('', [], null)).toBeNull()
  })

  it('defaults to one task for this PC alone and four with a rental', () => {
    expect(targetFor([{ kind: 'local' }])).toEqual({ hosts: [{ kind: 'local' }], sessionsPerHost: 1, contextPerSession: null, stopRentalsAfter: false })
    const r = targetFor([{ kind: 'vast', rentalId: 1 }])
    expect(r.sessionsPerHost).toBe(4)
    expect(r.contextPerSession).toBe(65536)
    expect(targetFor([{ kind: 'local' }], { sessionsPerHost: 3 }).contextPerSession).toBe(65536)
    expect(targetFor([], undefined).hosts).toEqual(DEFAULT_BATCH_TARGET.hosts)
    expect(targetFor([{ kind: 'local' }], { stopRentalsAfter: true }).stopRentalsAfter).toBe(false)
  })

  it('shares a known saved context out over the tasks at once', () => {
    const rent = [{ kind: 'vast', rentalId: 1 }] as BatchHost[]
    const local = [{ kind: 'local' }] as BatchHost[]
    // 86016 only fits one task of 65536: the saved context unchanged.
    expect(targetFor(rent, undefined, 86016)).toMatchObject({ sessionsPerHost: 1, contextPerSession: 86016 })
    expect(targetFor(rent, undefined, 262144)).toMatchObject({ sessionsPerHost: 4, contextPerSession: 65536 })
    expect(targetFor(rent, undefined, 196608)).toMatchObject({ sessionsPerHost: 3, contextPerSession: 65536 })
    expect(targetFor(rent, undefined, 131072)).toMatchObject({ sessionsPerHost: 2, contextPerSession: 65536 })
    // Tasks the user sets re-divide the default, rounded down to 1024, never over the total.
    expect(targetFor(rent, { sessionsPerHost: 4 }, 86016)).toMatchObject({ sessionsPerHost: 4, contextPerSession: 21504 })
    expect(targetFor(rent, { sessionsPerHost: 3 }, 100000)).toMatchObject({ contextPerSession: 32768 })
    // A typed context wins.
    expect(targetFor(rent, { contextPerSession: 40000 }, 262144)).toMatchObject({ sessionsPerHost: 4, contextPerSession: 40000 })
    // This PC: one task, the model's own context; several tasks share the total.
    expect(targetFor(local, undefined, 86016)).toMatchObject({ sessionsPerHost: 1, contextPerSession: null })
    expect(targetFor(local, { sessionsPerHost: 2 }, 131072)).toMatchObject({ sessionsPerHost: 2, contextPerSession: 65536 })
    // Within the lane limit: sixteen hosts allow one task each.
    const many = (n: number): BatchHost[] => Array.from({ length: n }, (_, i) => ({ kind: 'vast', rentalId: i + 1 }))
    expect(targetFor(many(16), undefined, 1_000_000).sessionsPerHost).toBe(1)
    // Unknown (null, 0, undefined) = today's defaults.
    for (const t of [null, 0, undefined]) expect(targetFor(rent, undefined, t)).toMatchObject({ sessionsPerHost: 4, contextPerSession: 65536 })
  })

  it("knows the context to share out: the RE load's, else the saved default load's", () => {
    const cfg = (c: number | null): LoadConfig => ({ ...DEFAULT_LOAD_CONFIG, contextLength: c })
    expect(knownReContext(cfg(86016), { config: cfg(32768), custom: true })).toEqual({ total: 86016, from: 're' })
    expect(knownReContext(null, { config: cfg(131072), custom: true })).toEqual({ total: 131072, from: 'default' })
    expect(knownReContext(cfg(null), { config: cfg(65536), custom: true })).toEqual({ total: 65536, from: 'default' })
    // A small default context was saved for other uses; a small RE one was saved for this.
    expect(knownReContext(null, { config: cfg(32768), custom: true })).toBeNull()
    expect(knownReContext(cfg(32768), null)).toEqual({ total: 32768, from: 're' })
    expect(knownReContext(null, { config: cfg(131072), custom: false })).toBeNull()
    expect(knownReContext(null, { config: cfg(null), custom: true })).toBeNull()
    expect(knownReContext(undefined, null)).toBeNull()
  })

  it('warns, without blocking, when a task gets little context', () => {
    expect(contextWarning({ contextPerSession: 21504 })).toBe('21,504 tokens per task is little for reverse engineering; 65,536 or more works much better.')
    expect(contextWarning({ contextPerSession: 65536 })).toBeNull()
    expect(contextWarning({ contextPerSession: null })).toBeNull()
    const choices = hostChoices('vast/q', models('vast/q'), true)
    expect(targetProblems(targetFor([{ kind: 'vast', rentalId: 1 }], { sessionsPerHost: 4 }, 86016), choices)).toEqual([])
  })

  it('clamps tasks at once by the lane limits', () => {
    expect(maxSessions(1)).toBe(8)
    expect(maxSessions(2)).toBe(8)
    expect(maxSessions(4)).toBe(4)
    expect(maxSessions(16)).toBe(1)
    expect(targetFor([{ kind: 'vast', rentalId: 1 }, { kind: 'vast', rentalId: 2 }, { kind: 'vast', rentalId: 3 }, { kind: 'vast', rentalId: 4 }], { sessionsPerHost: 8 }).sessionsPerHost).toBe(4)
  })

  it('says why a target cannot start', () => {
    const choices = hostChoices('vast/q', models('vast/q'), true)
    const good = targetFor([{ kind: 'vast', rentalId: 1 }])
    expect(targetProblems(good, choices)).toEqual([])
    expect(targetProblems({ ...good, contextPerSession: null }, choices)).toContain('Set the context per session.')
    expect(targetProblems(targetFor([{ kind: 'local' }]), choices)[0]).toContain('rentals only')
    expect(targetProblems(targetFor([{ kind: 'vast', rentalId: 2 }]), choices)[0]).toContain('not running')
    expect(targetProblems(targetFor([{ kind: 'vast', rentalId: 99 }]), choices)[0]).toContain('no longer there')
  })

  it('asks for a new target when an old host cannot run the model any more', () => {
    const choices = hostChoices('vast/q', models('vast/q'), true)
    const old = targetFor([{ kind: 'vast', rentalId: 1 }])
    expect(resumeTarget(old, choices)).toEqual(old)
    expect(resumeTarget(targetFor([{ kind: 'vast', rentalId: 2 }]), choices)).toBeNull()
    expect(resumeTarget(targetFor([{ kind: 'vast', rentalId: 1 }, { kind: 'vast', rentalId: 2 }]), choices)).toBeNull()
    expect(resumeTarget(undefined, choices)).toBeNull()
  })

  it('writes the memory hint and the cost per solved', () => {
    expect(ramHint(1)).toBe("About 1.3 GB of this PC's memory per task at once, for REA.")
    expect(ramHint(3)).toContain('about 3.9 GB for 3')
    expect(costPerSolved(3, 6)).toBe(0.5)
    expect(costPerSolved(3, 0)).toBeNull()
    expect(costPerSolved(undefined, 4)).toBeNull()
  })
})

describe('loading a remote model from the picker', () => {
  it('loads where exactly one running rental has it set up', () => {
    expect(loadPlan('vast/q', [rental({ id: 7 })])).toEqual({ kind: 'load', rentalId: 7 })
  })

  it('sends the user to the vast.ai page otherwise, and never says rent', () => {
    const none = loadPlan('vast/q', [])
    expect(none.kind).toBe('go')
    const stopped = loadPlan('vast/q', [rental({ rentalStatus: 'stopped' })])
    expect(stopped).toMatchObject({ kind: 'go' })
    expect((stopped as { sentence: string }).sentence).toContain('No vast.ai instance is running')
    const notSetUp = loadPlan('vast/q', [rental({ models: [] })])
    expect((notSetUp as { sentence: string }).sentence).toContain('has this model set up')
    const two = loadPlan('vast/q', [rental({ id: 1 }), rental({ id: 2 })])
    expect((two as { sentence: string }).sentence).toContain('more than one rental')
  })
})

describe('remote model definitions', () => {
  it('checks a Hugging Face source', () => {
    const ok = { kind: 'hf' as const, repo: 'unsloth/Qwen3-GGUF', bucket: false, revision: '', file: 'Qwen3-Q4_K_M.gguf', mmproj: '' }
    expect(validateRemoteSource(ok)).toBeNull()
    expect(validateRemoteSource({ ...ok, repo: 'nonsense' })).toContain('owner/name')
    expect(validateRemoteSource({ ...ok, file: '' })).toContain('model file')
    expect(validateRemoteSource({ ...ok, file: 'a b.gguf' })).toContain('spaces')
    expect(validateRemoteSource({ ...ok, file: '../x.gguf' })).toContain('..')
    expect(validateRemoteSource({ ...ok, mmproj: 'a\\b.gguf' })).toContain('projector')
    expect(validateRemoteSource({ kind: 'local', modelId: '' })).toContain('Choose')
    expect(validateRemoteSource({ kind: 'local', modelId: 'a/b.gguf' })).toBeNull()
  })

  it('trims a source and defaults the revision to main', () => {
    expect(cleanRemoteSource({ kind: 'hf', repo: ' o/n ', bucket: true, revision: ' ', file: ' f.gguf ', mmproj: ' ' })).toEqual({
      kind: 'hf',
      repo: 'o/n',
      bucket: true,
      revision: 'main',
      file: 'f.gguf',
      mmproj: ''
    })
  })

  it('names a new model from its file', () => {
    expect(defaultModelName({ kind: 'hf', repo: 'o/n', bucket: false, revision: 'main', file: 'dir/Model-Q4_K_M-00001-of-00003.gguf', mmproj: '' }, [])).toBe('Model-Q4_K_M')
    expect(defaultModelName({ kind: 'hf', repo: 'o/n', bucket: false, revision: 'main', file: 'Model-Q4.gguf', mmproj: '' }, [])).toBe('Model-Q4')
    expect(defaultModelName({ kind: 'local', modelId: 'dir/Big.gguf' }, [{ id: 'dir/Big.gguf', name: 'Big model' }] as never)).toBe('Big model')
  })
})
