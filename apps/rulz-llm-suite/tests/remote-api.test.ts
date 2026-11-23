import { describe, expect, it } from 'vitest'
import { DEFAULT_OFFER_FILTER } from '@shared/remote'
import { SecretStore, type SecretsDeps } from '../src/main/remote/secrets'
import { VastApi, VastApiError, cleanOfferFilter, offerSearchBody, parseInstance, type FetchLike } from '../src/main/remote/vastApi'
import { cleanAutoStop, cleanSettings, isDockerImage, isEngineTag } from '../src/main/remote/rentals'

interface Call {
  method: string
  url: string
  headers: Record<string, string>
  body: unknown
}
type Reply = { status: number; body?: unknown; raw?: string } | Error

/** A fetch that answers from a script and records every request. Nothing touches the network. */
function fakeFetch(script: (call: Call, n: number) => Reply): { fetch: FetchLike; calls: Call[] } {
  const calls: Call[] = []
  const fetch: FetchLike = async (url, init) => {
    const call: Call = { method: init.method, url, headers: init.headers, body: init.body ? JSON.parse(init.body) : undefined }
    calls.push(call)
    const r = script(call, calls.length)
    if (r instanceof Error) throw r
    const text = r.raw ?? (r.body === undefined ? '' : JSON.stringify(r.body))
    return { status: r.status, text: async () => text }
  }
  return { fetch, calls }
}

function api(script: (call: Call, n: number) => Reply, key: string | null = 'sk-test-key-12345'): { api: VastApi; calls: Call[]; sleeps: number[] } {
  const f = fakeFetch(script)
  const sleeps: number[] = []
  return { api: new VastApi({ key: async () => key, fetch: f.fetch, sleep: async (ms) => void sleeps.push(ms) }), calls: f.calls, sleeps }
}

const rentArgs = { offerId: 42, image: 'img', diskGb: 60, label: 'rulz-llm-aabbccdd', onstart: 'echo hi' }

describe('VastApi basics', () => {
  it('sends the key as a bearer token with a plain user agent, and never puts it in an error', async () => {
    const t = api(() => ({ status: 500 }))
    await expect(t.api.currentUser()).rejects.toThrow(VastApiError)
    expect(t.calls[0].headers.Authorization).toBe('Bearer sk-test-key-12345')
    expect(t.calls[0].headers['User-Agent']).toBe('rulz-llm-suite')
    const err = await t.api.currentUser().catch((e: Error) => e)
    expect((err as Error).message).not.toContain('sk-test-key')
  })

  it('makes no request without a key', async () => {
    const t = api(() => ({ status: 200, body: {} }), null)
    await expect(t.api.sshKeys()).rejects.toMatchObject({ kind: 'no-key' })
    expect(t.calls).toHaveLength(0)
  })

  it('reads the balance from credit or balance', async () => {
    expect((await api(() => ({ status: 200, body: { credit: '12.5' } })).api.currentUser()).credit).toBe(12.5)
    expect((await api(() => ({ status: 200, body: { balance: 3 } })).api.currentUser()).credit).toBe(3)
    expect((await api(() => ({ status: 200, body: { credit: 'abc' } })).api.currentUser()).credit).toBeNull()
  })
})

describe('offer search', () => {
  it('asks for on-demand amd64 offers with the disk size and filters total VRAM on the answer', async () => {
    const offer = (id: number, ram: number, n: number, extra: object = {}): object => ({ id, machine_id: id, gpu_name: 'RTX 4090', num_gpus: n, gpu_ram: ram, dph_total: 0.5, reliability2: 0.99, inet_down: 900, ...extra })
    const t = api(() => ({ status: 200, body: { offers: [offer(1, 24564, 1), offer(2, 12000, 1), offer(3, 24000, 1, { dph_total: 'x' }), offer(4, 24000, 1, { inet_down_cost: 0.01, duration: 86400 * 3 })] } }))
    const found = await t.api.searchOffers({ ...DEFAULT_OFFER_FILTER, diskGb: 80 })
    expect(found.map((o) => o.id)).toEqual([1, 4])
    expect(found[1].inetDownPerGb).toBe(0.01)
    expect(found[1].maxDays).toBe(3)
    const body = t.calls[0].body as Record<string, unknown>
    expect(t.calls[0].method).toBe('POST')
    expect(t.calls[0].url).toBe('https://console.vast.ai/api/v0/bundles/')
    expect(body.type).toBe('ondemand')
    expect(body.allocated_storage).toBe(80)
    expect(body.cpu_arch).toEqual({ eq: 'amd64' })
    expect(body.order).toEqual([['dph_total', 'asc']])
    expect(found[0].dlperf).toBeNull()
  })

  it('asks vast.ai for the best offers by the chosen column', async () => {
    const order = (sort: unknown): unknown => offerSearchBody(cleanOfferFilter({ ...DEFAULT_OFFER_FILTER, sort })).order
    expect(order('value')).toEqual([['dlperf_per_dphtotal', 'desc']])
    expect(order('vram')).toEqual([['gpu_total_ram', 'desc']])
    expect(order('reliability')).toEqual([['reliability', 'desc']])
    expect(order('down')).toEqual([['inet_down', 'desc']])
    expect(order('constructor')).toEqual([['dph_total', 'asc']])
  })

  it('finds one offer by its id, with the same filter, and tries twice', async () => {
    const one = { id: 7, gpu_name: 'RTX 4090', num_gpus: 1, gpu_ram: 24564, dph_total: 0.5, dlperf: 100 }
    const t = api((_c, n) => ({ status: 200, body: { offers: n < 2 ? [] : [{ ...one, id: 8 }, one] } }))
    const filter = { ...DEFAULT_OFFER_FILTER, diskGb: 90, sort: 'value' as const }
    expect(await t.api.findOffer(7, filter)).toMatchObject({ id: 7, dollarsPerHour: 0.5, dlperf: 100 })
    expect(t.calls).toHaveLength(2)
    expect(t.calls[1].body).toEqual({ ...offerSearchBody(filter), ask_contract_id: { eq: 7 } })
    const gone = api(() => ({ status: 200, body: { offers: [{ ...one, id: 8 }] } }))
    expect(await gone.api.findOffer(7, filter)).toBeNull()
    expect(gone.calls).toHaveLength(2)
  })

  it('accepts a single offer object and clamps filter values', async () => {
    const t = api(() => ({ status: 200, body: { offers: { id: 9, gpu_name: 'A100 SXM4', num_gpus: 1, gpu_ram: 40000, dph_total: 1 } } }))
    expect(await t.api.searchOffers({ ...DEFAULT_OFFER_FILTER, minVramGb: 30 })).toHaveLength(1)
    const f = cleanOfferFilter({ minVramGb: -5, maxGpus: 999, diskGb: 1e12, minReliability: 7, maxDollarsPerHour: 1e9, gpuName: 'x"; drop' })
    expect(f).toMatchObject({ minVramGb: 1, maxGpus: 16, diskGb: 4000, minReliability: 1, maxDollarsPerHour: 100, gpuName: '' })
    expect((offerSearchBody({ ...DEFAULT_OFFER_FILTER, gpuName: 'RTX 4090' }).gpu_name as { eq: string }).eq).toBe('RTX 4090')
  })

  it('backs off 1, 2, 4, 8 seconds on 429 for reads, then gives up', async () => {
    const t = api(() => ({ status: 429 }))
    await expect(t.api.searchOffers(DEFAULT_OFFER_FILTER)).rejects.toMatchObject({ kind: 'rate' })
    expect(t.sleeps).toEqual([1000, 2000, 4000, 8000])
    expect(t.calls).toHaveLength(5)
  })

  it('recovers when a 429 clears', async () => {
    const t = api((_c, n) => (n < 3 ? { status: 429 } : { status: 200, body: { offers: [] } }))
    expect(await t.api.searchOffers(DEFAULT_OFFER_FILTER)).toEqual([])
    expect(t.sleeps).toEqual([1000, 2000])
  })
})

describe('rent', () => {
  it('sends the documented body and returns the contract id', async () => {
    const t = api(() => ({ status: 200, body: { success: true, new_contract: 777 } }))
    expect(await t.api.rent(rentArgs)).toEqual({ contractId: 777 })
    expect(t.calls[0].method).toBe('PUT')
    expect(t.calls[0].url).toBe('https://console.vast.ai/api/v0/asks/42/')
    expect(t.calls[0].body).toEqual({ image: 'img', disk: 60, label: 'rulz-llm-aabbccdd', runtype: 'ssh_direct', target_state: 'running', cancel_unavail: true, onstart: 'echo hi', env: {} })
  })

  it('treats a 429 as refused and never retries the PUT', async () => {
    const t = api(() => ({ status: 429 }))
    await expect(t.api.rent(rentArgs)).rejects.toMatchObject({ definite: true, kind: 'rate' })
    expect(t.calls).toHaveLength(1)
    expect(t.sleeps).toEqual([])
  })

  it('does not retry after a timeout or network error and calls it ambiguous', async () => {
    const t = api(() => new Error('socket hang up'))
    await expect(t.api.rent(rentArgs)).rejects.toMatchObject({ definite: false, kind: 'network' })
    expect(t.calls).toHaveLength(1)
  })

  it('times out and reports it as ambiguous', async () => {
    const f: FetchLike = () => new Promise(() => undefined)
    const a = new VastApi({ key: async () => 'k', fetch: f })
    const original = global.setTimeout
    // Make the 30 s timer fire at once without waiting.
    global.setTimeout = ((fn: () => void, _ms?: number) => original(fn, 0)) as typeof setTimeout
    try {
      await expect(a.rent(rentArgs)).rejects.toMatchObject({ definite: false, kind: 'network' })
    } finally {
      global.setTimeout = original
    }
  })

  it('splits answers into refused and ambiguous', async () => {
    const cases: [Reply, boolean][] = [
      [{ status: 400, body: { msg: 'bad offer' } }, true],
      [{ status: 200, body: { success: false, msg: 'no longer available' } }, true],
      [{ status: 401, body: {} }, true],
      [{ status: 500, body: { x: 1 } }, false],
      [{ status: 502, raw: '<html>' }, false],
      [{ status: 200, raw: 'not json' }, false],
      [{ status: 200, body: { success: true } }, false],
      [{ status: 404, raw: 'nope' }, false]
    ]
    for (const [reply, definite] of cases) {
      const err = (await api(() => reply).api.rent(rentArgs).catch((e: unknown) => e)) as VastApiError
      expect(err).toBeInstanceOf(VastApiError)
      expect(err.definite, JSON.stringify(reply)).toBe(definite)
    }
  })
})

describe('instances', () => {
  const raw = {
    id: 5,
    label: ' mine ',
    actual_status: 'running',
    gpu_name: 'RTX 4090',
    num_gpus: 2,
    gpu_ram: 24564,
    dph_total: 0.8,
    disk_space: 100,
    storage_cost: 0.292,
    start_date: 1_700_000_000.5,
    public_ipaddr: ' 1.2.3.4\n',
    ports: { '22/tcp': [{ HostIp: '0.0.0.0', HostPort: '41022' }] },
    ssh_host: 'ssh4.vast.ai',
    ssh_port: 12345,
    status_msg: 'ok'
  }

  it('parses routes, cost while stopped and the start time', () => {
    const i = parseInstance(raw)!
    expect(i.routes).toEqual([
      { route: 'direct', host: '1.2.3.4', port: 41022 },
      { route: 'proxy', host: 'ssh4.vast.ai', port: 12345 }
    ])
    expect(i.storageDollarsPerHour).toBeCloseTo((0.292 * 100) / 730)
    expect(i.startedAt).toBe(1_700_000_000_500)
    expect(i.label).toBe('mine')
  })

  it('has no direct route for the array-of-int form of ports and rejects odd hosts and ports', () => {
    expect(parseInstance({ ...raw, ports: [22, 80] })!.routes.map((r) => r.route)).toEqual(['proxy'])
    expect(parseInstance({ ...raw, ssh_host: 'a b;rm', ssh_port: 70000, ports: {} })!.routes).toEqual([])
    expect(parseInstance({ ...raw, id: -1 })).toBeNull()
    expect(parseInstance({ ...raw, storage_cost: undefined })!.storageDollarsPerHour).toBeNull()
  })

  it('shows one instance and treats a 404 as none', async () => {
    expect((await api(() => ({ status: 200, body: { instances: raw } })).api.showInstance(5))!.id).toBe(5)
    expect(await api(() => ({ status: 404, body: { error: 'no' } })).api.showInstance(5)).toBeNull()
    expect(await api(() => ({ status: 200, body: { instances: { ...raw, id: 6 } } })).api.showInstance(5)).toBeNull()
  })

  it('pages through the whole list and fails when a page fails', async () => {
    const t = api((c) => {
      if (!c.url.includes('after_token')) return { status: 200, body: { instances: [{ ...raw, id: 1 }, { ...raw, id: 2 }], next_token: 'abc' } }
      return { status: 200, body: { instances: [{ ...raw, id: 3 }] } }
    })
    expect((await t.api.listInstances()).map((i) => i.id)).toEqual([1, 2, 3])
    expect(t.calls[0].url).toBe('https://console.vast.ai/api/v1/instances/?limit=25')
    expect(t.calls[1].url).toContain('after_token=abc')
    const bad = api((c) => (c.url.includes('after_token') ? { status: 500 } : { status: 200, body: { instances: [{ ...raw, id: 1 }], next_token: 'abc' } }))
    await expect(bad.api.listInstances()).rejects.toThrow(VastApiError)
  })

  it('retries start, stop and destroy on 429 and sends the documented bodies', async () => {
    const t = api((_c, n) => (n === 1 ? { status: 429 } : { status: 200, body: { success: true } }))
    await t.api.startInstance(5)
    expect(t.calls[1]).toMatchObject({ method: 'PUT', url: 'https://console.vast.ai/api/v0/instances/5/', body: { state: 'running' } })
    const s = api(() => ({ status: 200, body: { success: true } }))
    await s.api.stopInstance(5)
    await s.api.destroyInstance(5)
    expect(s.calls[0].body).toEqual({ state: 'stopped' })
    expect(s.calls[1].method).toBe('DELETE')
    await expect(api(() => ({ status: 200, body: { success: false, msg: 'busy' } })).api.stopInstance(5)).rejects.toThrow(/busy/)
  })

  it('reads ssh keys under key or public_key and posts a new one', async () => {
    const t = api(() => ({ status: 200, body: [{ key: 'ssh-ed25519 AAA a' }, { public_key: 'ssh-rsa BBB b' }, { id: 3 }] }))
    expect(await t.api.sshKeys()).toEqual(['ssh-ed25519 AAA a', 'ssh-rsa BBB b'])
    const p = api(() => ({ status: 200, body: { success: true } }))
    await p.api.addSshKey(' ssh-ed25519 AAA a \n')
    expect(p.calls[0]).toMatchObject({ method: 'POST', url: 'https://console.vast.ai/api/v0/ssh/', body: { ssh_key: 'ssh-ed25519 AAA a' } })
  })
})

describe('settings validation', () => {
  it('accepts the default image and rejects shell characters', () => {
    expect(isDockerImage('vastai/base-image:@vastai-automatic-tag')).toBe(true)
    expect(isDockerImage('nvidia/cuda:12.8.0-devel-ubuntu22.04')).toBe(true)
    expect(isDockerImage('x; rm -rf /')).toBe(false)
    expect(isDockerImage('Upper/Case')).toBe(false)
    expect(isEngineTag('b12407')).toBe(true)
    expect(isEngineTag('b12/../x')).toBe(false)
    expect(() => cleanSettings({ engineTag: 'latest', image: 'a/b' })).toThrow(/b12407/)
  })

  it('checks auto-stop ranges', () => {
    const base = { enabled: true, idleMinutes: 30, capHours: 12, apiCountsAsUse: false }
    expect(cleanAutoStop({ idleMinutes: 5, capHours: null }, base)).toMatchObject({ idleMinutes: 5, capHours: null })
    for (const bad of [{ idleMinutes: 4 }, { idleMinutes: 1441 }, { idleMinutes: 5.5 }, { capHours: 0 }, { capHours: 721 }, { enabled: 'yes' }]) {
      expect(() => cleanAutoStop(bad, base)).toThrow()
    }
  })
})

describe('SecretStore', () => {
  function store(over: Partial<SecretsDeps> = {}, files: Record<string, string> = {}): { s: SecretStore; written: unknown[]; files: Record<string, string> } {
    const written: unknown[] = []
    const deps: SecretsDeps = {
      available: () => true,
      encrypt: (p) => Buffer.from(`enc:${p}`),
      decrypt: (b) => b.toString().replace(/^enc:/, ''),
      readFile: async (p) => files[p] ?? null,
      writeFile: async (p, data) => {
        written.push(data)
        files[p] = JSON.stringify(data)
      },
      env: {},
      homedir: '/home/u',
      join: (...p) => p.join('/'),
      filePath: '/data/remote-secrets.json',
      ...over
    }
    return { s: new SecretStore(deps), written, files }
  }

  it('prefers a saved key, then the environment, then the CLI files', async () => {
    const t = store({ env: { VAST_API_KEY: 'envkey-1234567' } }, { '/home/u/.vast_api_key': 'filekey-1234567\n' })
    expect(await t.s.vastKey()).toEqual({ value: 'envkey-1234567', source: 'VAST_API_KEY environment variable' })
    await t.s.setVastKey('savedkey-12345')
    expect((await t.s.vastKey())?.source).toBe('saved in this app')
    await t.s.setVastKey('')
    expect((await t.s.vastKey())?.source).toBe('VAST_API_KEY environment variable')
    const f = store({ env: {} }, { '/home/u/.vast_api_key': 'filekey-1234567\n' })
    expect((await f.s.vastKey())?.value).toBe('filekey-1234567')
    const x = store({ env: { XDG_CONFIG_HOME: '/xdg' } }, { '/xdg/vastai/vast_api_key': 'xdgkey-1234567', '/home/u/.config/vastai/vast_api_key': 'other-1234567' })
    expect((await x.s.vastKey())?.value).toBe('xdgkey-1234567')
  })

  it('reads the Hugging Face token from HF_TOKEN, HF_HOME or the default cache', async () => {
    expect((await store({ env: { HF_TOKEN: 'hf_envtoken123' } }).s.hfToken())?.source).toBe('HF_TOKEN environment variable')
    expect((await store({ env: { HF_HOME: '/hf' } }, { '/hf/token': 'hf_homeToken1' }).s.hfToken())?.value).toBe('hf_homeToken1')
    expect((await store({}, { '/home/u/.cache/huggingface/token': 'hf_cacheTok12' }).s.hfToken())?.value).toBe('hf_cacheTok12')
    expect(await store().s.hfToken()).toBeNull()
  })

  it('refuses to save without encryption and never writes plain text', async () => {
    const t = store({ available: () => false })
    await expect(t.s.setVastKey('plainkey-12345')).rejects.toThrow(/encrypt/)
    expect(t.written).toHaveLength(0)
    const ok = store()
    await ok.s.setVastKey('savedkey-12345')
    expect(JSON.stringify(ok.written)).not.toContain('savedkey-12345')
    expect(ok.written[0]).toEqual({ vast: Buffer.from('enc:savedkey-12345').toString('base64') })
  })

  it('keeps the other secret when one is saved or forgotten', async () => {
    const t = store()
    await t.s.setVastKey('vastkey-123456')
    await t.s.setHfToken('hf_token-123456')
    await t.s.setVastKey('')
    expect(Object.keys(t.written.at(-1) as object)).toEqual(['hf'])
    expect((await t.s.hfToken())?.value).toBe('hf_token-123456')
  })

  it('scrubs every secret it has handed out, and other known long ones', async () => {
    const t = store({ env: { VAST_API_KEY: 'envkey-1234567', HF_TOKEN: 'hf_envtoken123' } }, { '/home/u/.vast_api_key': 'filekey-1234567' })
    await t.s.vastKey()
    await t.s.hfToken()
    expect(t.s.scrub('a envkey-1234567 b hf_envtoken123 c filekey-1234567 d short')).toBe('a *** b *** c *** d short')
    await t.s.setVastKey('savedkey-12345')
    expect(t.s.scrub('x savedkey-12345')).toBe('x ***')
  })
})
