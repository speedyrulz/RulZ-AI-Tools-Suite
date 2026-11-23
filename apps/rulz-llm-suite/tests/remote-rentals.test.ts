import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { DEFAULT_AUTO_STOP, DEFAULT_OFFER_FILTER, type RemoteModelDef, type RemoteOffer, type RentRequest } from '@shared/remote'
import type { ProbeResult, RentalOps, Secrets } from '../src/main/remote/contract'
import { remoteHandlers } from '../src/main/remote/ipc'
import { RentalManager, type RentalManagerDeps } from '../src/main/remote/rentals'
import { VastApiError, type VastApiClient, type VastInstanceDetail } from '../src/main/remote/vastApi'

const START = 1_700_000_000_000
const MIN = 60_000
const HOUR = 3_600_000
const BILLING_CALLS = ['rent', 'start', 'stop', 'destroy']
const PUB = 'ssh-ed25519 AAAAkey me@pc'

const model: RemoteModelDef = { id: 'vast/m1', name: 'M1', createdAt: 0, source: { kind: 'hf', repo: 'a/b', bucket: false, revision: 'main', file: 'm.gguf', mmproj: '' } }

function offer(over: Partial<RemoteOffer> = {}): RemoteOffer {
  return {
    id: 42,
    machineId: 1,
    gpuName: 'RTX 4090',
    numGpus: 1,
    gpuRamMiB: 24564,
    cpuCores: 16,
    cpuRamMiB: 64000,
    dollarsPerHour: 0.5,
    reliability: 0.99,
    inetDownMbps: 900,
    inetUpMbps: 500,
    inetDownPerGb: null,
    inetUpPerGb: null,
    cudaMaxGood: 12.9,
    geolocation: 'DE',
    maxDays: null,
    ...over
  }
}

function inst(id: number, over: Partial<VastInstanceDetail> = {}): VastInstanceDetail {
  return {
    id,
    label: `rulz-llm-${String(id).padStart(8, '0').slice(-8)}`,
    status: 'running',
    statusMessage: '',
    gpuName: 'RTX 4090',
    numGpus: 1,
    gpuRamMiB: 24564,
    dollarsPerHour: 0.5,
    storageDollarsPerHour: 0.01,
    diskGb: 60,
    startedAt: null,
    image: 'img',
    intended: 'running',
    routes: [{ route: 'direct', host: '1.2.3.4', port: 4022 }],
    ...over
  }
}

class Harness {
  t = START
  calls: string[] = []
  rentBodies: { offerId: number; label: string; onstart: string }[] = []
  instances = new Map<number, VastInstanceDetail>()
  offers: RemoteOffer[] = [offer()]
  accountKeys = [PUB]
  rentImpl: (r: { offerId: number; label: string }) => Promise<{ contractId: number }> = async () => ({ contractId: 777 })
  listFails = false
  listCalls = 0
  probeResult: ProbeResult = { gpus: [], diskFreeBytes: 200e9, foreignProcesses: false, engines: ['b12407'], watchdog: { armed: true }, models: [] }
  probeFails = false
  missing: number | null = 0
  inUseCalls: boolean[] = []
  stopping: number[] = []
  armed: unknown[] = []
  changes = 0
  dir = mkdtempSync(join(tmpdir(), 'rulz-rent-'))
  statePath = join(this.dir, 'remote.json')
  m!: RentalManager

  api: VastApiClient = {
    searchOffers: async () => {
      this.calls.push('search')
      return this.offers
    },
    rent: async (r) => {
      this.calls.push('rent')
      this.rentBodies.push(r)
      return this.rentImpl(r)
    },
    showInstance: async (id) => {
      this.calls.push('show')
      return this.instances.get(id) ?? null
    },
    listInstances: async () => {
      this.calls.push('list')
      this.listCalls++
      if (this.listFails) throw new VastApiError('vast.ai had an error (500).', 'server', false, 500)
      return [...this.instances.values()]
    },
    startInstance: async (id) => {
      this.calls.push('start')
      const i = this.instances.get(id)
      if (i) i.status = 'loading'
    },
    stopInstance: async (id) => {
      this.calls.push('stop')
      const i = this.instances.get(id)
      if (i) i.status = 'exited'
    },
    destroyInstance: async (id) => {
      this.calls.push('destroy')
      this.instances.delete(id)
    },
    sshKeys: async () => {
      this.calls.push('sshKeys')
      return this.accountKeys
    },
    addSshKey: async () => {
      this.calls.push('addSshKey')
    },
    currentUser: async () => {
      this.calls.push('user')
      return { credit: 5 }
    }
  }

  ops: RentalOps = {
    probe: async () => {
      if (this.probeFails) throw new Error('ssh refused')
      return this.probeResult
    },
    armWatchdog: async (_id, limits) => {
      this.armed.push(limits)
      return { armed: true }
    },
    onstartScript: () => '#!/bin/bash\necho onstart',
    installEngine: async () => undefined,
    transferModel: async () => undefined,
    removeModelFiles: async () => undefined,
    missingBytes: async () => this.missing,
    touch: async () => undefined,
    setInUse: (_id, v) => void this.inUseCalls.push(v),
    disconnect: () => undefined,
    log: () => ['ops line secretkey-1234']
  }

  secrets: Secrets = {
    vastKey: async () => ({ value: 'secretkey-1234', source: 'test' }),
    hfToken: async () => null,
    setVastKey: async () => undefined,
    setHfToken: async () => undefined,
    scrub: (s) => s.split('secretkey-1234').join('***')
  }

  constructor(over: Partial<RentalManagerDeps> = {}) {
    let n = 0
    this.m = new RentalManager({
      api: this.api,
      ops: this.ops,
      secrets: this.secrets,
      models: { list: () => [model] },
      statePath: this.statePath,
      localEngineBuild: async () => null,
      sshKey: async () => ({ available: true, publicKeyPath: 'k.pub', publicKey: PUB }),
      now: () => this.t,
      onChange: () => void this.changes++,
      onStopping: async (id) => void this.stopping.push(id),
      newLabel: () => `rulz-llm-${(++n).toString(16).padStart(8, '0')}`,
      ...over
    })
  }

  /** The manager is built again over the same remote.json (an app restart). */
  restart(over: Partial<RentalManagerDeps> = {}): RentalManager {
    const old = this.m
    old.dispose()
    const h = new Harness(over)
    h.t = this.t
    this.m = new RentalManager({ ...(h.m as unknown as { deps: RentalManagerDeps }).deps, api: this.api, ops: this.ops, statePath: this.statePath, now: () => this.t })
    return this.m
  }

  req(over: Partial<RentRequest> = {}): RentRequest {
    return { offerId: 42, shownDollarsPerHour: 0.5, filter: { ...DEFAULT_OFFER_FILTER }, autoStop: { ...DEFAULT_AUTO_STOP }, modelId: '', ...over }
  }

  async step(ms: number): Promise<void> {
    this.t += ms
    await this.m.tick()
    await this.m.idle()
  }

  billing(): string[] {
    return this.calls.filter((c) => BILLING_CALLS.includes(c))
  }

  state(): ReturnType<RentalManager['state']> {
    return this.m.state()
  }

  /** Rents (contract 777), lets vast report it running and the bring-up finish. */
  async running(over: Partial<RentRequest> = {}): Promise<number> {
    this.instances.set(777, inst(777, { label: 'rulz-llm-00000001' }))
    await this.m.rent(this.req(over))
    await this.step(10_000)
    expect(this.state().rentals[0].rentalStatus).toBe('running')
    return 777
  }

  /** An attached instance already running on the account. */
  async attached(id = 900): Promise<number> {
    this.instances.set(id, inst(id, { label: 'someone-elses' }))
    await this.m.attach(id)
    await this.step(10_000)
    return id
  }
}

const rental = (h: Harness, id: number) => h.state().rentals.find((r) => r.id === id)!

describe('rent', () => {
  it('writes the intent to disk before the PUT, then keeps only the rental', async () => {
    const h = new Harness()
    let onDisk: { intents: { label: string; offerId: number; state: string }[] } | null = null
    h.rentImpl = async (r) => {
      onDisk = JSON.parse(readFileSync(h.statePath, 'utf8'))
      expect(r.label).toMatch(/^rulz-llm-[0-9a-f]{8}$/)
      return { contractId: 777 }
    }
    h.instances.set(777, inst(777, { status: 'loading', label: 'rulz-llm-00000001' }))
    const s = await h.m.rent(h.req({ modelId: 'vast/m1' }))
    expect(onDisk!.intents).toHaveLength(1)
    expect(onDisk!.intents[0]).toMatchObject({ offerId: 42, state: 'sending' })
    expect(s.intents).toHaveLength(0)
    expect(s.rentals).toHaveLength(1)
    expect(s.rentals[0]).toMatchObject({ id: 777, origin: 'app', rentalStatus: 'starting' })
    const saved = JSON.parse(readFileSync(h.statePath, 'utf8'))
    expect(saved.intents).toHaveLength(0)
    expect(saved.rentals[0].pendingModelId).toBe('vast/m1')
    expect(h.rentBodies[0].onstart).toContain('onstart')
  })

  it('refuses any real rise above the confirmed price and accepts the same or a lower one', async () => {
    const h = new Harness()
    h.offers = [offer({ dollarsPerHour: 0.51 })]
    await expect(h.m.rent(h.req())).rejects.toThrow(/rose from \$0\.500 to \$0\.510/)
    expect(h.calls).not.toContain('rent')
    expect(h.state().intents).toHaveLength(0)
    h.offers = [offer({ dollarsPerHour: 0.501 })]
    await expect(h.m.rent(h.req())).rejects.toThrow(/rose/)
    expect(h.calls).not.toContain('rent')
    h.offers = [offer({ dollarsPerHour: 0.49 })]
    h.instances.set(777, inst(777, { label: 'rulz-llm-00000001' }))
    await h.m.rent(h.req())
    expect(h.calls).toContain('rent')
  })

  it('refuses without a key, without an ssh key on the account, and when the offer is gone', async () => {
    const h = new Harness()
    h.secrets.vastKey = async () => null
    await expect(h.m.rent(h.req())).rejects.toThrow(/API key/)
    h.secrets.vastKey = async () => ({ value: 'secretkey-1234', source: 'test' })
    h.accountKeys = ['ssh-ed25519 OTHER x']
    await expect(h.m.rent(h.req())).rejects.toThrow(/Add ssh key/)
    h.accountKeys = ['ssh-ed25519 AAAAkey another-comment']
    h.offers = []
    await expect(h.m.rent(h.req())).rejects.toThrow(/not available/)
    expect(h.calls).not.toContain('rent')
  })

  it('removes the intent after a definite refusal and allows another rent', async () => {
    const h = new Harness()
    h.rentImpl = async () => {
      throw new VastApiError('vast.ai refused the rent: gone.', 'client', true, 400)
    }
    await expect(h.m.rent(h.req())).rejects.toThrow(/Nothing was rented/)
    expect(h.state().intents).toHaveLength(0)
    expect(JSON.parse(readFileSync(h.statePath, 'utf8')).intents).toHaveLength(0)
    h.rentImpl = async () => ({ contractId: 777 })
    h.instances.set(777, inst(777, { label: 'rulz-llm-00000002' }))
    await h.m.rent(h.req())
    expect(h.state().rentals).toHaveLength(1)
  })

  it('never sends the PUT twice after a timeout, blocks other rents, and finds the rental by label', async () => {
    const h = new Harness()
    h.rentImpl = async () => {
      throw new VastApiError('vast.ai did not answer in time.', 'network', false)
    }
    const s = await h.m.rent(h.req({ modelId: 'vast/m1' }))
    expect(s.intents[0].state).toBe('checking')
    await expect(h.m.rent(h.req())).rejects.toThrow(/still open/)
    await h.step(1000)
    expect(h.listCalls).toBe(0)
    h.instances.set(31337, inst(31337, { label: 'rulz-llm-00000001', status: 'loading' }))
    await h.step(5000)
    expect(h.calls.filter((c) => c === 'rent')).toHaveLength(1)
    const after = h.state()
    expect(after.intents).toHaveLength(0)
    expect(after.rentals[0]).toMatchObject({ id: 31337, origin: 'app', label: 'rulz-llm-00000001' })
    expect(JSON.parse(readFileSync(h.statePath, 'utf8')).rentals[0].pendingModelId).toBe('vast/m1')
    await h.step(5 * MIN)
    expect(h.calls.filter((c) => c === 'rent')).toHaveLength(1)
  })

  it('settles an intent left by a crash by paging the listing, and keeps it open 10 minutes when not found', async () => {
    const h = new Harness()
    writeFileSync(
      h.statePath,
      JSON.stringify({ version: 1, rentals: [], settings: { engineTag: '', image: 'vastai/base-image:@vastai-automatic-tag' }, intents: [{ label: 'rulz-llm-deadbeef', offerId: 42, dollarsPerHour: 0.5, createdAt: START - 5000, state: 'sending', message: '', autoStop: DEFAULT_AUTO_STOP, modelId: '', diskGb: 60 }] })
    )
    await h.m.load()
    expect(h.state().intents[0].state).toBe('checking')
    for (let i = 0; i < 12; i++) await h.step(5000)
    expect(h.state().intents[0].state).toBe('checking')
    // Found on the next check.
    h.instances.set(555, inst(555, { label: 'rulz-llm-deadbeef', status: 'loading' }))
    await h.step(60_000)
    expect(h.state().intents).toHaveLength(0)
    expect(h.state().rentals[0].id).toBe(555)
    expect(h.calls).not.toContain('rent')
  })

  it('goes unresolved after 10 minutes, blocks renting until dismissed, and checkIntent can still find it', async () => {
    const h = new Harness()
    h.rentImpl = async () => {
      throw new VastApiError('vast.ai had an error (502).', 'server', false, 502)
    }
    await h.m.rent(h.req())
    for (let s = 0; s < 620; s += 5) await h.step(5000)
    const [intent] = h.state().intents
    expect(intent.state).toBe('unresolved')
    await expect(h.m.rent(h.req())).rejects.toThrow(/still open/)
    await expect(h.m.dismissIntent('rulz-llm-ffffffff')).resolves.toBeTruthy()
    h.instances.set(600, inst(600, { label: intent.label, status: 'loading' }))
    const s = await h.m.checkIntent(intent.label)
    expect(s.intents).toHaveLength(0)
    expect(s.rentals[0].id).toBe(600)
    expect(h.calls.filter((c) => c === 'rent')).toHaveLength(1)
  })

  it('can dismiss an unresolved intent, after which renting works again', async () => {
    const h = new Harness()
    h.rentImpl = async () => {
      throw new Error('weird')
    }
    await h.m.rent(h.req())
    for (let s = 0; s < 620; s += 5) await h.step(5000)
    await expect(h.m.dismissIntent(h.state().intents[0].label)).resolves.toBeTruthy()
    expect(h.state().intents).toHaveLength(0)
    h.rentImpl = async () => ({ contractId: 777 })
    h.instances.set(777, inst(777, { label: 'rulz-llm-00000002' }))
    await h.m.rent(h.req())
    expect(h.state().rentals).toHaveLength(1)
  })

  it('reports a contract price more than 2% above the one shown', async () => {
    const h = new Harness()
    h.instances.set(777, inst(777, { label: 'rulz-llm-00000001', dollarsPerHour: 0.52, status: 'loading' }))
    const s = await h.m.rent(h.req())
    expect(s.error).toMatch(/0\.520 per hour, more than the \$0\.500 shown/)
    expect(h.m.log(777).join('\n')).toMatch(/more than the \$0\.500 shown/)
    const quiet = new Harness()
    quiet.instances.set(777, inst(777, { label: 'rulz-llm-00000001', dollarsPerHour: 0.505, status: 'loading' }))
    expect((await quiet.m.rent(quiet.req())).error).toBeNull()
  })

  it('does not rent while the app starts, and a restart finishes the pending setup', async () => {
    const h = new Harness()
    await h.running({ modelId: 'vast/m1' })
    await h.step(1000)
    expect(rental(h, 777).models.find((m) => m.modelId === 'vast/m1')?.state).toBe('ready')
    expect(JSON.parse(readFileSync(h.statePath, 'utf8')).rentals[0].pendingModelId).toBe('')
    h.calls.length = 0
    const m2 = h.restart()
    await m2.load()
    await h.step(10_000)
    expect(h.billing()).toEqual([])
  })
})

describe('listing and status', () => {
  it('never marks a rental stopped or gone because the API failed', async () => {
    const h = new Harness()
    await h.running()
    h.listFails = true
    for (let i = 0; i < 6; i++) await h.step(MIN)
    const s = h.state()
    expect(s.rentals[0].rentalStatus).toBe('running')
    expect(s.error).toMatch(/error \(500\)/)
    h.listFails = false
    await h.step(MIN)
    expect(h.state().error).toBeNull()
  })

  it('marks a rental gone after three misses of a complete listing, and keeps it until detached', async () => {
    const h = new Harness()
    await h.running()
    h.instances.clear()
    await h.step(MIN)
    await h.step(MIN)
    expect(rental(h, 777).rentalStatus).toBe('running')
    await h.step(MIN)
    expect(rental(h, 777).rentalStatus).toBe('gone')
    expect(h.m.info(777)).toBeNull()
    await h.m.detach(777)
    expect(h.state().rentals).toHaveLength(0)
  })

  it('does not count a miss when the instance shows by id', async () => {
    const h = new Harness()
    await h.running()
    const i = h.instances.get(777)!
    h.api.listInstances = async () => {
      h.listCalls++
      return []
    }
    for (let k = 0; k < 4; k++) await h.step(MIN)
    expect(rental(h, 777).rentalStatus).toBe('running')
    expect(i.id).toBe(777)
  })

  it('refreshes every 60 seconds, every 10 while starting, and not at all with no rentals', async () => {
    const h = new Harness()
    await h.step(5 * MIN)
    expect(h.listCalls).toBe(0)
    await h.running()
    h.listCalls = 0
    await h.step(30_000)
    expect(h.listCalls).toBe(0)
    await h.step(40_000)
    expect(h.listCalls).toBe(1)
  })

  it('tells that an instance reported exited after renting is not starting', async () => {
    const h = new Harness()
    h.instances.set(777, inst(777, { label: 'rulz-llm-00000001', status: 'exited' }))
    await h.m.rent(h.req())
    await h.step(10_000)
    expect(rental(h, 777).rentalStatus).toBe('starting')
    await h.step(60_000)
    expect(h.state().error).toMatch(/does not look like it will start/)
    expect(rental(h, 777).rentalStatus).toBe('stopped')
    expect(h.billing()).toEqual(['rent'])
  })

  it('adds up the cost of the running time', async () => {
    const h = new Harness()
    await h.running()
    const before = rental(h, 777).spent
    await h.step(HOUR)
    expect(rental(h, 777).spent - before).toBeCloseTo(0.5, 2)
  })
})

describe('start, stop, destroy', () => {
  it('stops after re-reading the instance, refuses while in use unless forced, and cancels use first', async () => {
    const h = new Harness()
    const id = await h.running()
    const release = h.m.use(id, 'chat', 'window')
    await expect(h.m.stop(id, false)).rejects.toThrow(/in use \(chat\)/)
    expect(h.calls).not.toContain('stop')
    const calls = h.calls.length
    await h.m.stop(id, true)
    expect(h.calls.slice(calls)).toEqual(['show', 'stop'])
    expect(h.stopping).toEqual([id])
    expect(rental(h, id).rentalStatus).toBe('stopping')
    release()
    await h.step(MIN)
    expect(rental(h, id).rentalStatus).toBe('stopped')
  })

  it('bounds a stuck onStopping hook', async () => {
    const h = new Harness({ onStopping: () => new Promise(() => undefined), stopHookTimeoutMs: 20 })
    const id = await h.running()
    h.m.use(id, 'chat', 'window')
    await h.m.stop(id, true)
    expect(h.calls).toContain('stop')
  })

  it('starts only from a stopped state and restarts the cap and the cost', async () => {
    const h = new Harness()
    const id = await h.running()
    await expect(h.m.start(id)).rejects.toThrow(/stopped/)
    await h.m.stop(id, false)
    await h.step(MIN)
    expect(rental(h, id).rentalStatus).toBe('stopped')
    await h.step(2 * HOUR)
    const spent = rental(h, id).spent
    await h.m.start(id)
    expect(rental(h, id).rentalStatus).toBe('starting')
    expect(rental(h, id).spent).toBeLessThan(spent)
    expect(rental(h, id).capAt).toBe(h.t + 12 * HOUR)
    expect(h.calls.filter((c) => c === 'start')).toHaveLength(1)
  })

  it('refuses to destroy an attached instance, a wrong label, and a rental in use; names the label', async () => {
    const h = new Harness()
    const att = await h.attached()
    await expect(h.m.destroy(att, 'someone-elses', true)).rejects.toThrow(/app rented/)
    const id = await h.running()
    await expect(h.m.destroy(id, 'rulz-llm-wrong000', false)).rejects.toThrow(/does not match/)
    h.instances.get(id)!.label = 'renamed-on-site'
    await expect(h.m.destroy(id, 'rulz-llm-00000001', false)).rejects.toThrow(/not the one the app made/)
    h.instances.get(id)!.label = 'rulz-llm-00000001'
    const release = h.m.use(id, 'batch run', 'window')
    await expect(h.m.destroy(id, 'rulz-llm-00000001', false)).rejects.toThrow(/in use \(batch run\)/)
    expect(h.calls).not.toContain('destroy')
    await h.m.destroy(id, 'rulz-llm-00000001', true)
    expect(h.calls).toContain('destroy')
    expect(h.stopping).toContain(id)
    expect(h.state().rentals.map((r) => r.id)).toEqual([att])
    release()
  })

  it('does not act on an id it does not use', async () => {
    const h = new Harness()
    h.instances.set(5, inst(5))
    await expect(h.m.stop(5, true)).rejects.toThrow(/does not use/)
    await expect(h.m.destroy(5, 'x', true)).rejects.toThrow(/does not use/)
    expect(h.billing()).toEqual([])
  })
})

describe('uses', () => {
  it('counts window uses always and api uses only when the setting is on', async () => {
    const h = new Harness()
    const id = await h.running()
    h.inUseCalls.length = 0
    const api = h.m.use(id, 'api chat', 'api')
    expect(h.inUseCalls).toEqual([])
    expect(rental(h, id).inUse).toEqual(['api chat'])
    api()
    const win = h.m.use(id, 'chat', 'window')
    expect(h.inUseCalls).toEqual([true])
    win()
    win()
    expect(h.inUseCalls).toEqual([true, false])
    await h.m.setAutoStop(id, { apiCountsAsUse: true })
    const phone = h.m.use(id, 'phone', 'phone')
    expect(h.inUseCalls.at(-1)).toBe(true)
    phone()
    expect(h.inUseCalls.at(-1)).toBe(false)
  })

  it('updates lastUsedAt at the start and end of a use and during it', async () => {
    const h = new Harness()
    const id = await h.running()
    await h.step(5 * MIN)
    const end = h.m.use(id, 'chat', 'window')
    expect(rental(h, id).lastUsedAt).toBe(h.t)
    await h.step(10 * MIN)
    expect(rental(h, id).lastUsedAt).toBe(h.t)
    expect(rental(h, id).idleStopAt).toBeNull()
    await h.step(10 * MIN)
    end()
    expect(rental(h, id).idleStopAt).toBe(h.t + 30 * MIN)
  })
})

describe('auto-stop', () => {
  it('stops an idle app rental after a 60 second countdown', async () => {
    const h = new Harness()
    const id = await h.running()
    await h.step(31 * MIN)
    const cd = rental(h, id).countdown
    expect(cd?.reason).toMatch(/idle for 30 minutes/)
    expect(cd!.at).toBe(h.t + 60_000)
    expect(h.calls).not.toContain('stop')
    await h.step(30_000)
    expect(h.calls).not.toContain('stop')
    await h.step(30_000)
    expect(h.calls.filter((c) => c === 'stop')).toHaveLength(1)
    expect(h.stopping).toEqual([id])
    expect(rental(h, id).rentalStatus).toBe('stopping')
  })

  it('Keep running cancels an idle countdown and counts as use now', async () => {
    const h = new Harness()
    const id = await h.running()
    await h.step(31 * MIN)
    expect(rental(h, id).countdown).not.toBeNull()
    await h.m.keepRunning(id)
    expect(rental(h, id).countdown).toBeNull()
    expect(rental(h, id).lastUsedAt).toBe(h.t)
    await h.step(2 * MIN)
    expect(h.calls).not.toContain('stop')
    expect(rental(h, id).idleStopAt).toBeGreaterThan(h.t)
  })

  it('a use that starts during the countdown cancels it', async () => {
    const h = new Harness()
    const id = await h.running()
    await h.step(31 * MIN)
    h.m.use(id, 'chat', 'window')
    expect(rental(h, id).countdown).toBeNull()
    await h.step(5 * MIN)
    expect(h.calls).not.toContain('stop')
  })

  it('does not count API requests as use by default, so an idle rental still stops', async () => {
    const h = new Harness()
    const id = await h.running()
    h.m.use(id, 'api chat', 'api')
    await h.step(31 * MIN)
    expect(rental(h, id).countdown).not.toBeNull()
    await h.step(MIN)
    expect(h.calls).toContain('stop')
    const g = new Harness()
    const gid = await g.running({ autoStop: { ...DEFAULT_AUTO_STOP, apiCountsAsUse: true } })
    g.m.use(gid, 'api chat', 'api')
    await g.step(40 * MIN)
    expect(g.calls).not.toContain('stop')
  })

  it('never stops an idle rental while a foreign process is on the GPU, and does not nag', async () => {
    const h = new Harness()
    const id = await h.running()
    h.probeResult = { ...h.probeResult, foreignProcesses: true }
    await h.step(31 * MIN)
    await h.step(MIN)
    expect(h.calls).not.toContain('stop')
    expect(rental(h, id).countdown).toBeNull()
    await h.step(5 * MIN)
    expect(rental(h, id).countdown).toBeNull()
    expect(h.m.log(id).join('\n')).toMatch(/another process is using its GPU/)
  })

  it('when the GPU check fails, stops only rentals the app made', async () => {
    const h = new Harness()
    const id = await h.running()
    await h.m.setAutoStop(id, { idleMinutes: 5 })
    h.probeFails = true
    await h.step(6 * MIN)
    await h.step(MIN)
    expect(h.calls).toContain('stop')
    const a = new Harness()
    const att = await a.attached()
    await a.m.setAutoStop(att, { enabled: true, idleMinutes: 5 })
    a.probeFails = true
    await a.step(6 * MIN)
    await a.step(MIN)
    await a.step(MIN)
    expect(a.calls).not.toContain('stop')
  })

  it('stops at the hard cap even while in use, and Extend moves it', async () => {
    const h = new Harness()
    const id = await h.running({ autoStop: { ...DEFAULT_AUTO_STOP, capHours: 2 } })
    h.m.use(id, 'batch run', 'window')
    await h.step(2 * HOUR + MIN)
    expect(rental(h, id).countdown?.reason).toMatch(/time limit of 2 hours/)
    await h.m.extendCap(id, 1)
    expect(rental(h, id).countdown).toBeNull()
    expect(rental(h, id).capAt).toBeGreaterThan(h.t)
    await h.step(HOUR)
    expect(rental(h, id).countdown).not.toBeNull()
    await h.step(MIN)
    expect(h.calls).toContain('stop')
  })

  it('Keep running adds an hour to a cap countdown', async () => {
    const h = new Harness()
    const id = await h.running({ autoStop: { ...DEFAULT_AUTO_STOP, capHours: 1 } })
    await h.m.keepRunning(id)
    await h.step(HOUR + MIN)
    const before = rental(h, id).capAt!
    await h.m.keepRunning(id)
    expect(before).toBeLessThan(h.t)
    expect(rental(h, id).capAt).toBe(h.t + HOUR)
    expect(h.calls).not.toContain('stop')
  })

  it('an attached rental is never stopped at the cap while a foreign process runs, an app rental is', async () => {
    const h = new Harness()
    const att = await h.attached()
    await h.m.setAutoStop(att, { capHours: 1 })
    h.probeResult = { ...h.probeResult, foreignProcesses: true }
    await h.step(HOUR + 5 * MIN)
    await h.step(MIN)
    await h.step(MIN)
    expect(h.calls).not.toContain('stop')
    const a = new Harness()
    const id = await a.running({ autoStop: { ...DEFAULT_AUTO_STOP, capHours: 1 } })
    a.probeResult = { ...a.probeResult, foreignProcesses: true }
    a.m.use(id, 'chat', 'window')
    await a.step(HOUR + MIN)
    await a.step(MIN)
    expect(a.calls).toContain('stop')
  })

  it('retries a failed stop each minute and shows it', async () => {
    const h = new Harness()
    const id = await h.running()
    let fail = true
    const real = h.api.stopInstance
    h.api.stopInstance = async (i) => {
      if (fail) {
        h.calls.push('stop')
        throw new VastApiError('vast.ai had an error (500).', 'server', false, 500)
      }
      return real(i)
    }
    await h.step(31 * MIN)
    await h.step(MIN)
    expect(h.state().error).toMatch(/Could not stop it/)
    expect(rental(h, id).rentalStatus).toBe('running')
    fail = false
    await h.step(MIN)
    expect(rental(h, id).rentalStatus).toBe('stopping')
    expect(h.state().error).toBeNull()
  })

  it('stopAfterRun counts down like any stop, and is skipped for rentals in use or with foreign processes', async () => {
    const h = new Harness()
    const id = await h.running()
    h.m.stopAfterRun([id])
    expect(rental(h, id).countdown?.reason).toBe('The batch run ended')
    await h.step(61_000)
    expect(h.calls).toContain('stop')
    const b = new Harness()
    const bid = await b.running()
    b.m.use(bid, 'chat', 'window')
    b.m.stopAfterRun([bid])
    expect(rental(b, bid).countdown).toBeNull()
    expect(b.m.log(bid).join('\n')).toMatch(/in use \(chat\)/)
    const c = new Harness()
    const cid = await c.running()
    c.probeResult = { ...c.probeResult, foreignProcesses: true }
    await c.step(MIN)
    expect(rental(c, cid).foreignProcesses).toBe(true)
    c.m.stopAfterRun([cid])
    expect(rental(c, cid).countdown).toBeNull()
    expect(c.m.log(cid).join('\n')).toMatch(/another process/)
  })

  it('stopAfterRun re-checks the GPUs at the end of the countdown and does not stop', async () => {
    const h = new Harness()
    const id = await h.running()
    h.m.stopAfterRun([id])
    h.probeResult = { ...h.probeResult, foreignProcesses: true }
    await h.step(61_000)
    expect(h.calls).not.toContain('stop')
    expect(rental(h, id).countdown).toBeNull()
  })

  it('validates setAutoStop, persists it and re-arms the watchdog with the new limits', async () => {
    const h = new Harness()
    const id = await h.running()
    await expect(h.m.setAutoStop(id, { idleMinutes: 2 })).rejects.toThrow(/5 to 1440/)
    await expect(h.m.setAutoStop(id, { capHours: 0 })).rejects.toThrow(/1 to 720/)
    h.armed.length = 0
    await h.m.setAutoStop(id, { idleMinutes: 45, capHours: 6 })
    await h.m.idle()
    expect(rental(h, id).watchdog.state).toBe('armed')
    expect(h.armed[0]).toMatchObject({ origin: 'app', autoStop: { idleMinutes: 45, capHours: 6 } })
    expect(JSON.parse(readFileSync(h.statePath, 'utf8')).rentals[0].autoStop).toMatchObject({ idleMinutes: 45, capHours: 6 })
  })
})

describe('attach and setup', () => {
  it('attaches without arming a watchdog while auto-stop is off, and disarms one that was armed', async () => {
    const h = new Harness()
    h.probeResult = { ...h.probeResult, watchdog: { armed: false } }
    const id = await h.attached()
    expect(h.armed).toEqual([])
    expect(rental(h, id)).toMatchObject({ origin: 'attached', autoStop: { enabled: false, capHours: null } })
    expect(rental(h, id).watchdog.state).toBe('not-armed')
    const a = new Harness()
    a.probeResult = { ...a.probeResult, watchdog: { armed: true } }
    await a.attached()
    expect(a.armed).toHaveLength(1)
    expect(a.armed[0]).toMatchObject({ origin: 'attached', capAt: null, autoStop: { enabled: false } })
  })

  it('refuses to attach an instance that is not on the account or is already used', async () => {
    const h = new Harness()
    await expect(h.m.attach(123)).rejects.toThrow(/not on your vast.ai account/)
    const id = await h.attached()
    await expect(h.m.attach(id)).rejects.toThrow(/already/)
  })

  it('refuses a setup on an attached instance with too little disk, warns on an app rental', async () => {
    const h = new Harness()
    const att = await h.attached()
    h.missing = 196e9
    await h.m.setup(att, 'vast/m1')
    await h.m.idle()
    expect(rental(h, att).models[0].state).toBe('failed')
    expect(rental(h, att).models[0].error).toMatch(/Not enough free disk/)
    const a = new Harness()
    const id = await a.running()
    a.missing = 196e9
    await a.m.setup(id, 'vast/m1')
    await a.m.idle()
    expect(rental(a, id).models.find((m) => m.modelId === 'vast/m1')?.state).toBe('ready')
    expect(a.m.log(id).join('\n')).toMatch(/Warning: Not enough free disk/)
  })

  it('installs the engine when it is missing, throttles progress and counts setup as a use', async () => {
    const h = new Harness()
    const id = await h.running()
    h.probeResult = { ...h.probeResult, engines: [] }
    let during: string[] = []
    h.ops.installEngine = async (_i, tag, onProgress) => {
      expect(tag).toBe('b12407')
      during = rental(h, id).inUse
      const before = h.changes
      for (let i = 0; i < 100; i++) onProgress({ stage: 'x', bytesDone: i, bytesTotal: 100 })
      expect(h.changes - before).toBeLessThanOrEqual(1)
    }
    await h.m.setup(id, null)
    await h.m.idle()
    expect(during).toEqual(['setup'])
    expect(rental(h, id).engine).toMatchObject({ state: 'ready', tag: 'b12407' })
    expect(rental(h, id).inUse).toEqual([])
    await expect(h.m.setup(id, 'vast/nope')).rejects.toThrow(/not saved/)
  })

  it('cancels a setup through its AbortController', async () => {
    const h = new Harness()
    const id = await h.running()
    h.ops.transferModel = (_i, _d, _p, signal) => new Promise((_, reject) => signal.addEventListener('abort', () => reject(new Error('aborted'))))
    await h.m.setup(id, 'vast/m1')
    await new Promise((r) => setTimeout(r, 10))
    await h.m.cancelSetup(id)
    await h.m.idle()
    expect(rental(h, id).models.find((m) => m.modelId === 'vast/m1')).toMatchObject({ state: 'failed', error: 'Cancelled.' })
  })

  it('resolves the engine tag from settings, then this PC build, then the default; checks the settings', async () => {
    const h = new Harness({ localEngineBuild: async () => 9876 })
    await h.m.load()
    expect(h.state().resolvedEngineTag).toBe('b9876')
    await h.m.saveSettings({ engineTag: 'b12000', image: 'vastai/base-image:@vastai-automatic-tag' })
    expect(h.state().resolvedEngineTag).toBe('b12000')
    await expect(h.m.saveSettings({ engineTag: 'b1/../2', image: 'vastai/base-image:@vastai-automatic-tag' })).rejects.toThrow()
    await expect(h.m.saveSettings({ engineTag: '', image: 'img; rm' })).rejects.toThrow(/docker image/)
    expect(new Harness().state().resolvedEngineTag).toBe('b12407')
    const odd = new Harness({ localEngineBuild: async () => 12.5 })
    await odd.m.load()
    expect(odd.state().resolvedEngineTag).toBe('b12407')
  })
})

describe('money: only four methods bill', () => {
  it('attach, setup, use, ticks, refresh, keys and settings never call a billing API', async () => {
    const h = new Harness()
    const att = await h.attached()
    await h.m.setAutoStop(att, { enabled: true })
    await h.m.setup(att, 'vast/m1')
    await h.m.idle()
    const release = h.m.use(att, 'chat', 'window')
    h.m.use(att, 'api', 'api')()
    release()
    for (let i = 0; i < 20; i++) await h.step(10_000)
    await h.m.refresh()
    await h.m.searchOffers(DEFAULT_OFFER_FILTER)
    await h.m.instances()
    await h.m.addSshKey()
    await h.m.setVastKey('')
    await h.m.saveSettings({ engineTag: '', image: 'vastai/base-image:@vastai-automatic-tag' })
    await h.m.keepRunning(att)
    await h.m.detach(att).catch(() => undefined)
    expect(h.billing()).toEqual([])
  })

  it('a stopped rental is never started by a tick, a restart or a use', async () => {
    const h = new Harness()
    const id = await h.running()
    await h.m.stop(id, false)
    await h.step(MIN)
    h.calls.length = 0
    h.m.use(id, 'chat', 'window')
    const m2 = h.restart()
    await m2.load()
    for (let i = 0; i < 10; i++) await h.step(MIN)
    expect(h.billing()).toEqual([])
    expect(rental(h, id).rentalStatus).toBe('stopped')
  })

  it('the IPC table has the four billing channels and nothing registers the others as billing', async () => {
    const h = new Harness()
    const table = remoteHandlers(h.m)
    expect(Object.keys(table)).toEqual(expect.arrayContaining(['remote:rent', 'remote:start', 'remote:stop', 'remote:destroy']))
    for (const gone of ['remote:load', 'remote:estimate', 'remote:saveModel', 'remote:deleteModel']) expect(table).not.toHaveProperty(gone)
    // The read channels cannot bill.
    for (const ch of ['remote:state', 'remote:refresh', 'remote:instances', 'remote:log'] as const) {
      await (table[ch] as (...a: unknown[]) => unknown)(1)
    }
    expect(h.billing()).toEqual([])
  })
})

describe('quit', () => {
  it('summarises running rentals with when they stop themselves, and stops them all', async () => {
    const h = new Harness()
    const id = await h.running()
    const sum = h.m.quitSummary()
    expect(sum.running).toHaveLength(1)
    expect(sum.running[0]).toMatchObject({ label: 'rulz-llm-00000001', dollarsPerHour: 0.5 })
    expect(sum.running[0].stopsItself).toMatch(/after 30 idle minutes, at the latest at/)
    h.probeResult = { ...h.probeResult }
    const none = new Harness()
    const nid = await none.running()
    await none.m.setAutoStop(nid, { enabled: false, capHours: null })
    expect(none.m.quitSummary().running[0].stopsItself).toBeNull()
    expect(await h.m.stopAllForQuit()).toEqual([])
    expect(h.calls).toContain('stop')
    expect(h.stopping).toEqual([id])
    expect(h.m.quitSummary().running).toHaveLength(0)
  })

  it('reports an error instead of throwing and leaves an attached rental with a foreign process running', async () => {
    const h = new Harness()
    const att = await h.attached()
    h.probeResult = { ...h.probeResult, foreignProcesses: true }
    await h.step(MIN)
    const errs = await h.m.stopAllForQuit()
    expect(errs[0]).toMatch(/left running/)
    expect(h.calls).not.toContain('stop')
    expect(att).toBe(900)
  })

  it('dispose stops the timer and disconnects every rental', async () => {
    const dropped: number[] = []
    let timer: (() => void) | null = null
    const h = new Harness({ setTimer: (fn) => ((timer = fn), 'handle'), clearTimer: () => void (timer = null) })
    h.ops.disconnect = (id) => void dropped.push(id)
    await h.running()
    h.m.startTicking()
    expect(timer).not.toBeNull()
    h.m.dispose()
    expect(timer).toBeNull()
    expect(dropped).toContain(777)
  })
})

describe('logs and secrets', () => {
  it('merges ops lines with its own notes, scrubbed and capped at 500', async () => {
    const h = new Harness()
    const id = await h.running()
    for (let i = 0; i < 600; i++) h.m.note(id, `line ${i} secretkey-1234`)
    const log = h.m.log(id)
    expect(log.length).toBe(500)
    expect(log.join('\n')).not.toContain('secretkey-1234')
    expect(log.at(-1)).toMatch(/line 599 \*\*\*/)
  })

  it('ipc validates its input', async () => {
    const h = new Harness()
    const t = remoteHandlers(h.m)
    expect(() => t['remote:start'](-1 as number)).toThrow()
    expect(() => t['remote:stop'](1.5, false)).toThrow()
    expect(() => t['remote:stop'](1, 'yes' as unknown as boolean)).toThrow()
    expect(() => t['remote:destroy'](1, 'x'.repeat(500), true)).toThrow()
    expect(() => t['remote:setVastKey']('k'.repeat(1000))).toThrow()
    expect(() => t['remote:rent']({ ...h.req(), shownDollarsPerHour: Number.NaN })).toThrow()
    expect(() => t['remote:rent']({ ...h.req(), offerId: 1e30 })).toThrow()
    expect(() => t['remote:searchOffers'](null as never)).toThrow()
    const offers = await t['remote:searchOffers']({ ...DEFAULT_OFFER_FILTER, diskGb: 1e12 })
    expect(offers).toHaveLength(1)
  })
})

describe('review fixes', () => {
  /** Runs an app rental to its idle countdown. */
  async function idleCountdown(h: Harness): Promise<number> {
    const id = await h.running()
    await h.step(31 * MIN)
    expect(rental(h, id).countdown).not.toBeNull()
    return id
  }

  it('Keep running during the last probe still keeps it running', async () => {
    const h = new Harness()
    const id = await idleCountdown(h)
    h.ops.probe = async () => {
      await h.m.keepRunning(id)
      return h.probeResult
    }
    await h.step(2 * MIN)
    expect(h.calls).not.toContain('stop')
    expect(rental(h, id).rentalStatus).toBe('running')
  })

  it('a use that begins while models are unloaded for the stop cancels it', async () => {
    const h: Harness = new Harness({ onStopping: async (id) => void h.m.use(id, 'chat', 'window') })
    const id = await idleCountdown(h)
    await h.step(2 * MIN)
    expect(h.calls).not.toContain('stop')
    expect(rental(h, id).countdown).toBeNull()
  })

  it('a failed stop counts down again, and Keep running ends the retry', async () => {
    const h = new Harness()
    const id = await idleCountdown(h)
    const stop = h.api.stopInstance
    let fails = 0
    h.api.stopInstance = async () => {
      fails++
      throw new VastApiError('vast.ai had an error (500).', 'server', false, 500)
    }
    await h.step(2 * MIN)
    expect(fails).toBe(1)
    expect(rental(h, id).error).toMatch(/Could not stop it/)
    expect(rental(h, id).countdown).not.toBeNull()
    await h.m.keepRunning(id)
    expect(rental(h, id).error).toBeUndefined()
    h.api.stopInstance = stop
    await h.step(2 * MIN)
    expect(h.calls).not.toContain('stop')
  })

  it('a status that flaps keeps the cap; a real restart elsewhere begins a new period from now', async () => {
    const h = new Harness()
    const id = await h.running()
    await h.m.extendCap(id, 5)
    const cap = rental(h, id).capAt
    h.instances.get(id)!.startedAt = START - 100 * HOUR
    h.instances.get(id)!.status = 'exited'
    await h.step(2 * MIN)
    expect(rental(h, id).rentalStatus).toBe('stopped')
    h.instances.get(id)!.status = 'running'
    await h.step(2 * MIN)
    expect(rental(h, id).rentalStatus).toBe('running')
    expect(rental(h, id).capAt).toBe(cap)
    h.instances.get(id)!.status = 'exited'
    for (let i = 0; i < 11; i++) await h.step(2 * MIN)
    h.instances.get(id)!.status = 'running'
    await h.step(2 * MIN)
    expect(rental(h, id).rentalStatus).toBe('running')
    expect(rental(h, id).capAt).toBe(h.t + 12 * HOUR)
  })

  it('Extend gives the watchdog the new limits', async () => {
    const h = new Harness()
    const id = await h.running()
    const n = h.armed.length
    await h.m.extendCap(id, 3)
    await h.m.idle()
    expect(h.armed).toHaveLength(n + 1)
    expect((h.armed[n] as { capAt: number }).capAt).toBe(rental(h, id).capAt)
    expect(rental(h, id).watchdog.state).toBe('armed')
  })

  it('while the new limits are not on the instance, quit does not promise a stop time', async () => {
    const h = new Harness()
    const id = await h.running()
    h.ops.armWatchdog = async () => {
      throw new Error('ssh refused')
    }
    await h.m.extendCap(id, 3)
    expect(h.m.quitSummary().running[0].stopsItself).toBeNull()
    await h.m.idle()
    expect(h.m.quitSummary().running[0].stopsItself).toBeNull()
  })

  it('quit looks at an attached instance before stopping it, and leaves it when it cannot', async () => {
    const h = new Harness()
    await h.attached()
    h.probeResult = { ...h.probeResult, foreignProcesses: true }
    expect((await h.m.stopAllForQuit())[0]).toMatch(/another process/)
    h.probeFails = true
    expect((await h.m.stopAllForQuit())[0]).toMatch(/could not be checked/)
    expect(h.calls).not.toContain('stop')
  })

  it('a cap switched on for an attached instance counts from now', async () => {
    const h = new Harness()
    h.instances.set(900, inst(900, { label: 'x', startedAt: START - 72 * HOUR }))
    await h.m.attach(900)
    await h.step(10_000)
    await h.m.setAutoStop(900, { capHours: 12 })
    expect(rental(h, 900).capAt).toBe(h.t + 12 * HOUR)
  })

  it('a destroyed rental stays in the list until vast.ai no longer shows it', async () => {
    const h = new Harness()
    const id = await h.running()
    h.api.destroyInstance = async () => void h.calls.push('destroy')
    const s = await h.m.destroy(id, 'rulz-llm-00000001', false)
    expect(s.rentals[0].error).toMatch(/still shows/)
    expect(h.m.quitSummary().running).toHaveLength(1)
    h.instances.delete(id)
    await h.step(2 * MIN)
    expect(h.state().rentals).toHaveLength(0)
  })

  it('an answer without a price does not make the rental free, and the contract price is compared once it is known', async () => {
    const h = new Harness()
    h.instances.set(777, inst(777, { label: 'rulz-llm-00000001', dollarsPerHour: 0 }))
    await h.m.rent(h.req())
    expect(rental(h, 777).dollarsPerHour).toBe(0.5)
    h.instances.get(777)!.dollarsPerHour = 0.7
    await h.step(2 * MIN)
    expect(rental(h, 777).error).toContain('costs $0.700')
  })

  it('counts the time the app was closed when the rental is still running', async () => {
    const h = new Harness()
    const id = await h.running()
    await h.step(2 * MIN)
    const before = rental(h, id).spent
    h.t += 4 * HOUR
    h.restart()
    await h.m.load()
    await h.step(1000)
    expect(rental(h, id).spent - before).toBeGreaterThan(1.9)
    expect(rental(h, id).spent - before).toBeLessThan(2.2)
  })

  it('a rent left open by a crash can be dismissed after ten minutes even when the app was started again', async () => {
    const h = new Harness()
    h.rentImpl = async () => {
      throw new VastApiError('timeout', 'network', false)
    }
    await h.m.rent(h.req())
    const label = h.state().intents[0].label
    await expect(h.m.dismissIntent(label)).rejects.toThrow(/still checking/)
    h.t += 11 * MIN
    h.restart()
    await h.m.load()
    await expect(h.m.dismissIntent(label)).rejects.toThrow(/still checking/)
    await h.step(6000)
    await h.m.dismissIntent(label)
    expect(h.state().intents).toHaveLength(0)
  })

  it('an idle rental whose last look showed another GPU process is not stopped blind', async () => {
    const h = new Harness()
    const id = await h.running()
    h.probeResult = { ...h.probeResult, foreignProcesses: true }
    await h.step(2 * MIN)
    h.probeFails = true
    await h.step(31 * MIN)
    await h.step(2 * MIN)
    await h.step(2 * MIN)
    expect(h.calls).not.toContain('stop')
    expect(rental(h, id).rentalStatus).toBe('running')
  })

  it('a list that fails after a restart does not count the closed time twice', async () => {
    const h = new Harness()
    const id = await h.running()
    await h.step(2 * MIN)
    const before = rental(h, id).spent
    h.t += 4 * HOUR
    h.restart()
    await h.m.load()
    h.listFails = true
    for (let i = 0; i < 60; i++) await h.step(MIN)
    h.listFails = false
    await h.step(2 * MIN)
    expect(rental(h, id).spent - before).toBeGreaterThan(2.4)
    expect(rental(h, id).spent - before).toBeLessThan(2.65)
  })

  it('idle stop does not act while a GPU is busy with something nothing here uses', async () => {
    const h = new Harness()
    const id = await h.running()
    h.probeResult = { ...h.probeResult, gpus: [{ index: 0, name: 'x', totalMiB: 24000, usedMiB: 900, utilization: 80 }] }
    await h.step(31 * MIN)
    await h.step(2 * MIN)
    expect(h.calls).not.toContain('stop')
    h.probeResult = { ...h.probeResult, gpus: [{ index: 0, name: 'x', totalMiB: 24000, usedMiB: 900, utilization: 0 }] }
    await h.step(11 * MIN)
    await h.step(2 * MIN)
    expect(h.calls).toContain('stop')
    expect(rental(h, id).rentalStatus).toBe('stopping')
  })

  it('idle stop switched on for an attached instance waits out the idle time first', async () => {
    const h = new Harness()
    h.instances.set(900, inst(900, { label: 'x', startedAt: START - 72 * HOUR }))
    await h.m.attach(900)
    await h.step(10_000)
    await h.m.setAutoStop(900, { enabled: true, idleMinutes: 30 })
    await h.step(5 * MIN)
    expect(rental(h, 900).countdown).toBeNull()
    await h.step(26 * MIN)
    expect(rental(h, 900).countdown).not.toBeNull()
  })

  it('an older arming that finishes late does not put the old limit back', async () => {
    const h = new Harness()
    const id = await h.running()
    const seen: number[] = []
    let calls = 0
    let open: () => void = () => undefined
    const gate = new Promise<void>((r) => (open = r))
    h.ops.armWatchdog = async (_id, l) => {
      if (calls++ === 0) await gate
      seen.push(l.capAt as number)
      return { armed: true }
    }
    await h.m.extendCap(id, 1)
    await h.m.extendCap(id, 1)
    await new Promise((r) => setTimeout(r, 5))
    expect(rental(h, id).watchdog.state).toBe('unknown')
    open()
    await h.m.idle()
    expect(seen[seen.length - 1]).toBe(rental(h, id).capAt)
    expect(rental(h, id).watchdog.state).toBe('armed')
  })

  it('a search that could not read the account is not a check of a lost rent', async () => {
    const h = new Harness()
    h.rentImpl = async () => {
      throw new VastApiError('timeout', 'network', false)
    }
    await h.m.rent(h.req())
    const label = h.state().intents[0].label
    h.listFails = true
    for (let i = 0; i < 15; i++) await h.step(MIN)
    expect(h.state().intents[0].state).toBe('checking')
    await expect(h.m.dismissIntent(label)).rejects.toThrow(/still checking/)
  })
})
