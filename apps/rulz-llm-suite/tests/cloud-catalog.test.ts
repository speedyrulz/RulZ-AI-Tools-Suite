import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { MIN_CONTEXT_LIMIT, samplerSupport } from '@shared/cloud'
import fixture from './fixtures/openrouter-models.json'
import { CatalogStore, parseCachedInfo, parseCatalog, parseModel } from '../src/main/cloud/catalog'
import { CloudKeyStore } from '../src/main/cloud/keyStore'
import { CloudModelStore, clampContext, clampReply, toModelEntry } from '../src/main/cloud/models'
import { maxPriceFor, parsePrices, toMicro, worstCaseMicro } from '../src/main/cloud/prices'
import { estimateInput, estimateText } from '../src/main/cloud/tokens'
import { FakeNet, TEST_KEY, fakeSafeStorage, jsonResponse, pickInfo } from './helpers/cloudKit'

let dir: string
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'cloud-cat-'))
})
afterEach(() => rmSync(dir, { recursive: true, force: true }))

describe('catalog parser on the real fixture', () => {
  const models = parseCatalog(fixture)
  const slugs = models.map((m) => m.slug).sort()

  it('keeps only the usable models', () => {
    expect(slugs).toEqual(
      ['anthropic/claude-fable-5.2', 'baidu/ernie-5.5-flash-preview:free', 'google/gemini-4-neon', 'inclusionai/ling-3.2-flash:free', 'inference-net/schematron-v2-turbo', 'openai/gpt-6.2-astra-pro'].sort()
    )
  })

  it('leaves out routers, ~ aliases, :batch, image output and models with no max_tokens', () => {
    for (const bad of ['openrouter/auto', 'nvidia/switchyard', '~unbiased/pareto-latest', 'openai/gpt-6.2-astra-pro:batch', 'anthropic/claude-fable-5.2:batch', 'google/gemini-nano-banana-2.1', 'sakana/fugu-ultra-v2']) {
      expect(slugs).not.toContain(bad)
    }
  })

  it('reads prices in micro-dollars per token, with overrides and cache writes in the maximum', () => {
    const gpt = models.find((m) => m.slug === 'openai/gpt-6.2-astra-pro')!
    expect(gpt.prices.prompt).toBe(8)
    expect(gpt.prices.completion).toBe(40)
    // The 272k-token override is dearer: 16 and 60; cache write 10 / 20 stays below 16.
    expect(gpt.prices.promptMax).toBe(20)
    expect(gpt.prices.completionMax).toBe(60)
    const fable = models.find((m) => m.slug === 'anthropic/claude-fable-5.2')!
    expect(fable.prices.promptMax).toBe(16) // input_cache_write_1h
    expect(fable.prices.completionMax).toBe(40)
    const gem = models.find((m) => m.slug === 'google/gemini-4-neon')!
    expect(gem.prices.completionMax).toBe(15) // internal_reasoning equals completion
    expect(gem.prices.image).toBe(3)
  })

  it('reads capabilities', () => {
    const fable = models.find((m) => m.slug === 'anthropic/claude-fable-5.2')!
    expect(fable.vision).toBe(true)
    expect(fable.tools).toBe(true)
    expect(fable.free).toBe(false)
    expect(fable.maxReply).toBe(128000)
    expect(fable.reasoning).toEqual({ mandatory: true, efforts: ['max', 'xhigh', 'high', 'medium', 'low'], supportsMaxTokens: false })
    const ernie = models.find((m) => m.slug.startsWith('baidu/'))!
    expect(ernie.vision).toBe(false)
    expect(ernie.free).toBe(true)
    const ling = models.find((m) => m.slug.startsWith('inclusionai/'))!
    expect(ling.reasoning).toMatchObject({ mandatory: false })
    const schema = models.find((m) => m.slug.startsWith('inference-net/'))!
    expect(schema.reasoning).toBeNull()
    expect(schema.tools).toBe(false)
    expect(schema.free).toBe(false)
  })

  it('derives the samplers a model honours', () => {
    const fable = samplerSupport(models.find((m) => m.slug === 'anthropic/claude-fable-5.2')!)
    expect(fable.temperature).toBe(false)
    expect(fable.stop).toBe(true)
    expect(fable.thinking).toBe(true)
    expect(fable.thinkingOff).toBe(false)
    const gem = samplerSupport(models.find((m) => m.slug === 'google/gemini-4-neon')!)
    expect(gem.temperature).toBe(true)
    expect(gem.topP).toBe(true)
    expect(gem.topK).toBe(false)
  })

  it('refuses an unknown shape', () => {
    expect(() => parseCatalog({ nope: 1 })).toThrow()
    expect(parseModel(null)).toBeNull()
    expect(parseModel({ id: 'a/b' })).toBeNull()
  })

  it('round trips an info through the cache reader', () => {
    for (const m of models) expect(parseCachedInfo(JSON.parse(JSON.stringify(m)))).toEqual(m)
    expect(parseCachedInfo({ slug: 'x' })).toBeNull()
  })
})

describe('prices', () => {
  const base = { prompt: '0.000001', completion: '0.000002' }
  it('micro-dollars per token', () => {
    expect(parsePrices(base)).toMatchObject({ prompt: 1, completion: 2, promptMax: 1, completionMax: 2, request: 0, image: 0 })
    expect(parsePrices({ prompt: '0.00000003', completion: '0.00000015' })).toMatchObject({ prompt: 0.03, completion: 0.15 })
  })
  it('router and unparsable or negative prices are unusable', () => {
    expect(parsePrices({ prompt: '-1', completion: '-1' })).toBeNull()
    expect(parsePrices({ prompt: 'abc', completion: '1' })).toBeNull()
    expect(parsePrices({ prompt: '1', completion: '' })).toBeNull()
    expect(parsePrices({ prompt: '1', completion: '1', request: '-0.1' })).toBeNull()
    expect(parsePrices({ prompt: '1', completion: '1', web_search: 'x' })).toBeNull()
    expect(parsePrices({ prompt: '1', completion: '1', overrides: [{ prompt: '-2' }] })).toBeNull()
    expect(parsePrices({ prompt: '1', completion: '1', overrides: 'x' })).toBeNull()
    expect(parsePrices({ prompt: '1', completion: '1e999' })).toBeNull()
    expect(parsePrices({ completion: '1' })).toBeNull()
    expect(parsePrices(null)).toBeNull()
  })
  it('includes every override, cache write and internal reasoning in the maximum', () => {
    const p = parsePrices({
      prompt: '0.000001',
      completion: '0.000002',
      input_cache_write: '0.0000015',
      internal_reasoning: '0.000005',
      request: '0.01',
      image: '0.002',
      overrides: [{ min_prompt_tokens: 1000, prompt: '0.000004', completion: '0.000003', input_cache_write: '0.000008', internal_reasoning: '0.000009' }]
    })!
    expect(p.promptMax).toBe(8)
    expect(p.completionMax).toBe(9)
    expect(p.prompt).toBe(1)
    expect(p.completion).toBe(2)
    expect(p.request).toBe(10000)
    expect(p.image).toBe(2000)
  })
  it('toMicro rounds up without float noise', () => {
    expect(toMicro(0.0042)).toBe(4200)
    expect(toMicro(0.0000001)).toBe(1)
    expect(toMicro(0)).toBe(0)
    expect(toMicro(-1)).toBe(0)
    expect(toMicro(NaN)).toBe(0)
  })
  it('worst case = (input x 1.25 + 2000 per image) x prompt + max_tokens x completion + request + images x image, rounded up', () => {
    const p = parsePrices({ prompt: '0.000002', completion: '0.00001', request: '0.001', image: '0.0005' })!
    // (1000*1.25 + 2000*2) * 2 + 500*10 + 1000 + 2*500 = 10500 + 5000 + 1000 + 1000
    expect(worstCaseMicro(p, 1000, 2, 500)).toBe(17500)
    expect(worstCaseMicro(p, 1, 0, 0)).toBe(1000 + 3)
  })
  it('max_price: per million tokens, request and image in USD only when above 0', () => {
    const p = parsePrices({ prompt: '0.000002', completion: '0.00001', request: '0.001', image: '0.0005' })!
    const mp = maxPriceFor(p)
    expect(mp.prompt).toBeGreaterThanOrEqual(2)
    expect(mp.prompt).toBeLessThan(2.0001)
    expect(mp.completion).toBeGreaterThanOrEqual(10)
    expect(mp.request).toBeCloseTo(0.001, 10)
    expect(mp.image).toBeCloseTo(0.0005, 10)
    expect(maxPriceFor(parsePrices(base)!)).not.toHaveProperty('request')
    expect(maxPriceFor(parsePrices({ prompt: '0', completion: '0' })!)).toEqual({ prompt: 0, completion: 0 })
  })
})

describe('token estimate', () => {
  it('ASCII chars / 3.5, every other char 1', () => {
    expect(estimateText('')).toBe(0)
    expect(estimateText('a'.repeat(35))).toBe(10)
    expect(estimateText('日本語')).toBe(3)
    expect(estimateText('a'.repeat(7) + '日')).toBe(3)
    // An emoji (two UTF-16 units) is one character.
    expect(estimateText('😀')).toBe(1)
  })
  it('adds about 4 a message, counts images only by number, never data URLs', () => {
    const data = 'data:image/png;base64,' + 'A'.repeat(100000)
    const e = estimateInput([
      { role: 'user', content: [{ type: 'text', text: 'a'.repeat(35) }, { type: 'image_url', image_url: { url: data } }] },
      { role: 'assistant', content: 'a'.repeat(35) }
    ])
    expect(e.images).toBe(1)
    expect(e.tokens).toBe(10 + 10 + 8)
  })
  it('counts tool calls and tool definitions', () => {
    const e = estimateInput([{ role: 'assistant', content: null, tool_calls: [{ id: 'c', type: 'function', function: { name: 'abcdefg', arguments: '{"a":1}' } }] }], [{ type: 'function', function: { name: 't' } }])
    expect(e.tokens).toBeGreaterThan(8)
  })
})

describe('catalog store', () => {
  const path = (): string => join(dir, 'cloud-catalog.json')
  const answer = (): Response => jsonResponse(fixture)
  const make = (net: FakeNet, now: () => number) => new CatalogStore({ path: path(), fetch: net.fetch, now, scrub: (s) => s })

  it('fetches without a key, caches on disk, refreshes after 24 h or on request', async () => {
    let t = 1_000_000
    const net = new FakeNet(() => answer())
    const store = make(net, () => t)
    const r1 = await store.get(false)
    expect(r1.error).toBeNull()
    expect(r1.catalog.models).toHaveLength(6)
    expect(net.calls).toHaveLength(1)
    expect(net.calls[0].url).toBe('https://openrouter.ai/api/v1/models')
    expect(net.calls[0].headers.authorization).toBeUndefined()
    expect(net.calls[0].redirect).toBe('error')
    t += 3600_000
    await store.get(false)
    expect(net.calls).toHaveLength(1)
    // A new process reads the disk cache.
    const store2 = make(net, () => t)
    await store2.get(false)
    expect(net.calls).toHaveLength(1)
    await store2.get(true)
    expect(net.calls).toHaveLength(2)
    t += 25 * 3600_000
    await store2.get(false)
    expect(net.calls).toHaveLength(3)
  })

  it('offline: returns the stale cache with the error', async () => {
    let t = 1_000_000
    const net = new FakeNet(() => answer())
    const store = make(net, () => t)
    await store.get(false)
    t += 30 * 3600_000
    net.handler = () => new Error('getaddrinfo ENOTFOUND openrouter.ai')
    const r = await store.get(false)
    expect(r.catalog.models).toHaveLength(6)
    expect(r.error).toContain("model list")
    net.handler = () => new Response('no', { status: 500 })
    expect((await store.get(true)).error).toContain('500')
  })

  it('offline with no cache: an empty list and the error', async () => {
    const net = new FakeNet(() => new Error('down'))
    const r = await make(net, () => 5).get(false)
    expect(r.catalog).toEqual({ models: [], at: 0 })
    expect(r.error).toBeTruthy()
  })

  it('an unreadable cache file is ignored', async () => {
    writeFileSync(path(), '{not json')
    const net = new FakeNet(() => answer())
    const r = await make(net, () => 5).get(false)
    expect(r.catalog.models).toHaveLength(6)
  })
})

describe('model store', () => {
  const path = (): string => join(dir, 'cloud.json')
  const make = (): CloudModelStore => new CloudModelStore({ path: path(), now: () => 123 })

  it('defaults: $1 limit, training not allowed', async () => {
    const s = make()
    await s.load()
    expect(s.options()).toEqual({ dailyLimitUsd: 1, allowTraining: false })
    expect(s.list()).toEqual([])
  })

  it('adds with defaults min(context, 32768) and min(8192, provider max), persists, and makes entries', async () => {
    const s = make()
    await s.load()
    const def = await s.add(pickInfo(fixture, 'anthropic/claude-fable-5.2'))
    expect(def.contextLimit).toBe(32768)
    expect(def.replyLimit).toBe(8192)
    expect(def.allowApps).toBe(false)
    const small = await s.add(pickInfo(fixture, 'inference-net/schematron-v2-turbo'))
    expect(small.contextLimit).toBe(32768)
    expect(small.replyLimit).toBe(8192)
    await expect(s.add(pickInfo(fixture, 'anthropic/claude-fable-5.2'))).rejects.toThrow(/already/)
    const again = make()
    await again.load()
    expect(again.list().map((d) => d.slug)).toEqual(['anthropic/claude-fable-5.2', 'inference-net/schematron-v2-turbo'])
    const e = toModelEntry(def)
    expect(e).toMatchObject({ id: 'openrouter/anthropic/claude-fable-5.2', name: 'Anthropic: Claude Fable 5.2', path: '', files: [], sizeBytes: 0, info: null, folder: '' })
    expect(e.cloud?.provider).toBe('openrouter')
    expect(e.remote).toBeUndefined()
  })

  it('clamps limits on update', async () => {
    const s = make()
    await s.load()
    await s.add(pickInfo(fixture, 'inference-net/schematron-v2-turbo'))
    let d = await s.update('inference-net/schematron-v2-turbo', { contextLimit: 10_000_000, replyLimit: 10_000_000, allowApps: true })
    expect(d.contextLimit).toBe(128000)
    expect(d.replyLimit).toBe(8192)
    expect(d.allowApps).toBe(true)
    d = await s.update('inference-net/schematron-v2-turbo', { contextLimit: 5, replyLimit: -3 })
    expect(d.contextLimit).toBe(MIN_CONTEXT_LIMIT)
    expect(d.replyLimit).toBe(256)
    await expect(s.update('nope', {})).rejects.toThrow()
  })

  it('options: a limit is null or at least $0.01', async () => {
    const s = make()
    await s.load()
    await s.setOptions({ dailyLimitUsd: null })
    expect(s.options().dailyLimitUsd).toBeNull()
    await s.setOptions({ dailyLimitUsd: 0.01, allowTraining: true })
    expect(s.options()).toEqual({ dailyLimitUsd: 0.01, allowTraining: true })
    await expect(s.setOptions({ dailyLimitUsd: 0.001 })).rejects.toThrow()
    await expect(s.setOptions({ dailyLimitUsd: NaN })).rejects.toThrow()
    await expect(s.setOptions({ dailyLimitUsd: -1 })).rejects.toThrow()
    expect(s.options().dailyLimitUsd).toBe(0.01)
  })

  it('a damaged file never turns the limit off; bad entries are skipped', async () => {
    const good = pickInfo(fixture, 'anthropic/claude-fable-5.2')
    writeFileSync(
      path(),
      JSON.stringify({
        options: { dailyLimitUsd: 'lots', allowTraining: 'yes' },
        models: [{ slug: good.slug, info: good, contextLimit: 'x', replyLimit: 1e9, allowApps: 1, addedAt: 5 }, { slug: 'x' }, null, { slug: 'y', info: { slug: 'z' } }]
      })
    )
    const s = make()
    await s.load()
    expect(s.options()).toEqual({ dailyLimitUsd: 1, allowTraining: false })
    expect(s.list()).toHaveLength(1)
    expect(s.list()[0]).toMatchObject({ contextLimit: 32768, replyLimit: 128000, allowApps: false })
    writeFileSync(path(), '{broken')
    const s2 = make()
    await s2.load()
    expect(s2.options().dailyLimitUsd).toBe(1)
    expect(s2.list()).toEqual([])
  })

  it('removes', async () => {
    const s = make()
    await s.load()
    await s.add(pickInfo(fixture, 'anthropic/claude-fable-5.2'))
    await s.remove('anthropic/claude-fable-5.2')
    expect(s.list()).toEqual([])
    const s2 = make()
    await s2.load()
    expect(s2.list()).toEqual([])
  })

  it('refreshes info from a newer catalog and clamps again', async () => {
    const s = make()
    await s.load()
    await s.add(pickInfo(fixture, 'anthropic/claude-fable-5.2'))
    await s.update('anthropic/claude-fable-5.2', { contextLimit: 900_000 })
    const newer = pickInfo(fixture, 'anthropic/claude-fable-5.2', { contextLength: 200_000, name: 'Renamed' })
    expect(await s.refreshInfo([newer])).toBe(true)
    expect(s.list()[0].info.name).toBe('Renamed')
    expect(s.list()[0].contextLimit).toBe(200_000)
    expect(await s.refreshInfo([newer])).toBe(false)
    // A model gone from the catalog keeps its last info.
    expect(await s.refreshInfo([])).toBe(false)
    expect(s.list()).toHaveLength(1)
  })

  it('clamp helpers work on small models', () => {
    const info = pickInfo(fixture, 'inference-net/schematron-v2-turbo', { contextLength: 1000, maxReply: null })
    expect(clampContext(undefined, info)).toBe(1000)
    expect(clampReply(undefined, info)).toBe(1000)
  })
})

describe('key store', () => {
  const path = (): string => join(dir, 'cloud-secrets.json')

  it('saves encrypted, never as text, and loads again', async () => {
    const k = new CloudKeyStore({ path: path(), safeStorage: fakeSafeStorage() })
    await k.save(` ${TEST_KEY} `)
    expect(k.get()).toBe(TEST_KEY)
    expect(readFileSync(path(), 'utf8')).not.toContain(TEST_KEY)
    const k2 = new CloudKeyStore({ path: path(), safeStorage: fakeSafeStorage() })
    await k2.load()
    expect(k2.get()).toBe(TEST_KEY)
    await k2.clear()
    expect(k2.get()).toBeNull()
    const k3 = new CloudKeyStore({ path: path(), safeStorage: fakeSafeStorage() })
    await k3.load()
    expect(k3.get()).toBeNull()
  })

  it('refuses to save without encryption', async () => {
    const k = new CloudKeyStore({ path: path(), safeStorage: fakeSafeStorage(false) })
    await expect(k.save(TEST_KEY)).rejects.toThrow(/encrypt/)
    expect(k.get()).toBeNull()
    expect(() => readFileSync(path(), 'utf8')).toThrow()
    // The refused key is still scrubbed from text.
    expect(k.scrub(`bad ${TEST_KEY}`)).not.toContain(TEST_KEY)
  })

  it('reads no environment variable', async () => {
    const old = process.env.OPENROUTER_API_KEY
    process.env.OPENROUTER_API_KEY = TEST_KEY
    try {
      const k = new CloudKeyStore({ path: path(), safeStorage: fakeSafeStorage() })
      await k.load()
      expect(k.get()).toBeNull()
    } finally {
      if (old === undefined) delete process.env.OPENROUTER_API_KEY
      else process.env.OPENROUTER_API_KEY = old
    }
  })

  it('an undecryptable or damaged file is no key', async () => {
    writeFileSync(path(), JSON.stringify({ key: Buffer.from('garbage').toString('base64') }))
    const k = new CloudKeyStore({ path: path(), safeStorage: fakeSafeStorage() })
    await k.load()
    expect(k.get()).toBeNull()
    writeFileSync(path(), '{{')
    await k.load()
    expect(k.get()).toBeNull()
  })

  it('scrubs the key, a longer key containing it, bearer headers and key-shaped text', async () => {
    const k = new CloudKeyStore({ path: path(), safeStorage: fakeSafeStorage() })
    await k.save(TEST_KEY)
    expect(k.scrub(`x ${TEST_KEY} y ${TEST_KEY}`)).toBe('x *** y ***')
    expect(k.scrub('Authorization: Bearer abcdefghij')).not.toContain('abcdefghij')
    expect(k.scrub('seen sk-or-v1-zzzzzzzzzzzzzz')).not.toContain('zzzz')
  })
})
