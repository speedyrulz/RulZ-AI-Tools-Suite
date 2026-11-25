import { describe, expect, it } from 'vitest'
import { DEFAULT_DAILY_LIMIT_USD, MAX_CLOUD_SESSIONS, cloudModelId, type CloudModelDef, type CloudModelInfo, type CloudState } from '../src/shared/cloud'
import type { LoadedInstance } from '../src/shared/types'
import { pickActive } from '../src/renderer/src/lib/activeModel'
import {
  CATALOG_ROW_CAP,
  EMPTY_FILTER,
  badgesFor,
  batchCostHint,
  capRows,
  chipText,
  cleanLimit,
  clampInt,
  contextBounds,
  costText,
  expiresWarning,
  filterCatalog,
  isCloudId,
  modelSpend,
  priceLine,
  priceShort,
  replyBounds,
  spendView,
  spentTodayText,
  supportsOf
} from '../src/renderer/src/lib/cloud'
import { hostChoices, hostKey, maxSessions, targetFor, targetProblems } from '../src/renderer/src/lib/remote'
import { callKey, newer } from '../src/renderer/src/stores/cloud'

const info = (over: Partial<CloudModelInfo> = {}): CloudModelInfo => ({
  slug: 'acme/big',
  name: 'Big',
  description: '',
  contextLength: 200_000,
  maxReply: 16_000,
  prices: { promptMax: 3, completionMax: 15, request: 0, image: 0, prompt: 3, completion: 15 },
  free: false,
  vision: true,
  tools: true,
  reasoning: { mandatory: false, efforts: ['low', 'high'], supportsMaxTokens: true },
  supported: ['temperature', 'top_p', 'max_tokens', 'seed', 'tools', 'reasoning'],
  created: 100,
  expires: null,
  ...over
})

const state = (over: Partial<CloudState> = {}, spent = 0, reserved = 0, limit: number | null = 1): CloudState => ({
  rev: 1,
  keySaved: true,
  key: null,
  error: null,
  options: { dailyLimitUsd: limit, allowTraining: false },
  models: [],
  spend: { day: '2026-11-25', spent, reserved, models: {}, days: [] },
  spendError: null,
  ...over
})

describe('prices and badges', () => {
  it('shows prices per million tokens, or Free', () => {
    expect(priceLine(info())).toBe('$3.00 in · $15.00 out per million tokens')
    expect(priceShort(info())).toBe('$3.00 / $15.00')
    const free = info({ free: true, prices: { promptMax: 0, completionMax: 0, request: 0, image: 0, prompt: 0, completion: 0 } })
    expect(priceLine(free)).toBe('Free')
    expect(priceShort(free)).toBe('Free')
  })

  it('lists Free, Images, Tools and Thinking in that order, only what is true', () => {
    expect(badgesFor(info({ free: true })).map((b) => b.label)).toEqual(['Free', 'Images', 'Tools', 'Thinking'])
    expect(badgesFor(info({ vision: false, tools: false, reasoning: null })).map((b) => b.label)).toEqual([])
  })

  it('shows the cost of a reply only when it cost something', () => {
    expect(costText(undefined)).toBeNull()
    expect(costText(0)).toBeNull()
    expect(costText(4200)).toBe('$0.0042')
  })
})

describe('spending', () => {
  it('is fine, close, or reached, always with words', () => {
    expect(spendView(state({}, 100_000)).tone).toBe('ok')
    const close = spendView(state({}, 850_000))
    expect(close.tone).toBe('warn')
    expect(close.word).toMatch(/close/i)
    const over = spendView(state({}, 600_000, 400_000))
    expect(over.tone).toBe('danger')
    expect(over.word).toMatch(/refused/)
  })

  it('has no bar without a limit', () => {
    const v = spendView(state({}, 5_000_000, 0, null))
    expect(v.limit).toBeNull()
    expect(v.share).toBe(0)
    expect(v.tone).toBe('ok')
    expect(spentTodayText(state({}, 5_000_000, 0, null))).toBe('$5.00 today')
  })

  it('words today against the limit', () => {
    expect(spentTodayText(state({}, 120_000))).toBe('$0.12 of $1.00 today')
  })

  it('keeps a typed limit at cents and at least the minimum', () => {
    expect(cleanLimit(0)).toBe(0.01)
    expect(cleanLimit(2.345)).toBe(2.35)
    expect(cleanLimit(null)).toBe(DEFAULT_DAILY_LIMIT_USD)
    expect(cleanLimit(Number.NaN)).toBe(DEFAULT_DAILY_LIMIT_USD)
  })

  it('lists today by model, biggest first, without zeros', () => {
    const s = state()
    s.spend.models = { 'a/x': 5, 'b/y': 900, 'c/z': 0, 'a/w': 5 }
    expect(modelSpend(s).map((r) => r.slug)).toEqual(['b/y', 'a/w', 'a/x'])
  })

  it('shows the top-bar chip only with a key and spending, or a connected model', () => {
    expect(chipText(null, [])).toBeNull()
    expect(chipText(state({}, 0), [])).toBeNull()
    expect(chipText(state({ keySaved: false }, 500_000), [])).toBeNull()
    expect(chipText(state({}, 120_000), [])).toBe('OpenRouter $0.12 today')
    expect(chipText(state({}, 0), [{ cloud: { provider: 'openrouter', free: false } }])).toBe('OpenRouter $0.00 today')
  })

  it('tells a batch run what it may cost and the limit', () => {
    expect(batchCostHint(state({}, 300_000))).toContain('$0.30 of $1.00')
    expect(batchCostHint(state({}, 0, 0, null))).toContain('No daily limit')
  })
})

describe('limits of an added model', () => {
  const def = (over: Partial<CloudModelInfo> = {}): Pick<CloudModelDef, 'contextLimit' | 'info'> => ({ contextLimit: 32_768, info: info(over) })

  it('keeps the context between the minimum and the model context', () => {
    expect(contextBounds(info())).toEqual({ min: 2048, max: 200_000 })
    expect(contextBounds(info({ contextLength: 1000 }))).toEqual({ min: 1000, max: 1000 })
  })

  it('caps the reply at the provider maximum, else the context', () => {
    expect(replyBounds(def())).toEqual({ min: 256, max: 16_000 })
    expect(replyBounds(def({ maxReply: null, contextLength: 8000 }))).toEqual({ min: 256, max: 8000 })
    expect(replyBounds(def({ maxReply: 99_000, contextLength: 8000 }))).toEqual({ min: 256, max: 8000 })
  })

  it('clamps typed numbers', () => {
    expect(clampInt(10, 256, 1000)).toBe(256)
    expect(clampInt(5000.4, 256, 1000)).toBe(1000)
    expect(clampInt(500.6, 256, 1000)).toBe(501)
  })

  it('warns when OpenRouter drops the model soon, or already did', () => {
    const now = Date.parse('2026-11-25T00:00:00Z')
    expect(expiresWarning(info(), now)).toBeNull()
    expect(expiresWarning(info({ expires: '2027-06-01' }), now)).toBeNull()
    expect(expiresWarning(info({ expires: '2026-12-05' }), now)).toMatch(/drops this model on 2026-12-05 \(in 10 days\)/)
    expect(expiresWarning(info({ expires: '2026-11-01' }), now)).toMatch(/dropped/)
  })
})

describe('the catalog browser', () => {
  const catalog = [
    info({ slug: 'a/old', name: 'Old Model', created: 1, vision: false, tools: false, free: false }),
    info({ slug: 'b/new-free', name: 'New Free', created: 9, free: true, vision: false }),
    info({ slug: 'c/vision', name: 'Seer', created: 5, vision: true, tools: false })
  ]

  it('leaves out models already added, newest first', () => {
    const rows = filterCatalog(catalog, new Set(['c/vision']), EMPTY_FILTER)
    expect(rows.map((m) => m.slug)).toEqual(['b/new-free', 'a/old'])
  })

  it('needs every search word in the name or the slug', () => {
    expect(filterCatalog(catalog, new Set(), { ...EMPTY_FILTER, query: 'new free' }).map((m) => m.slug)).toEqual(['b/new-free'])
    expect(filterCatalog(catalog, new Set(), { ...EMPTY_FILTER, query: 'c/ seer' }).map((m) => m.slug)).toEqual(['c/vision'])
    expect(filterCatalog(catalog, new Set(), { ...EMPTY_FILTER, query: 'nothing' })).toEqual([])
  })

  it('applies every chip that is on', () => {
    expect(filterCatalog(catalog, new Set(), { ...EMPTY_FILTER, free: true }).map((m) => m.slug)).toEqual(['b/new-free'])
    expect(filterCatalog(catalog, new Set(), { ...EMPTY_FILTER, images: true }).map((m) => m.slug)).toEqual(['c/vision'])
    expect(filterCatalog(catalog, new Set(), { ...EMPTY_FILTER, tools: true }).map((m) => m.slug)).toEqual(['b/new-free'])
    expect(filterCatalog(catalog, new Set(), { ...EMPTY_FILTER, free: true, images: true })).toEqual([])
  })

  it('caps a long list and says how many are hidden', () => {
    const rows = Array.from({ length: CATALOG_ROW_CAP + 30 }, (_, i) => i)
    const { shown, hidden } = capRows(rows)
    expect(shown).toHaveLength(CATALOG_ROW_CAP)
    expect(hidden).toBe(30)
    expect(capRows([1, 2]).hidden).toBe(0)
  })
})

describe('what the controls honour', () => {
  it('derives support from the entry, and leaves other models alone', () => {
    expect(supportsOf(undefined)).toBeUndefined()
    expect(supportsOf({})).toBeUndefined()
    const s = supportsOf({ cloud: { info: info() } })!
    expect(s.temperature).toBe(true)
    expect(s.topK).toBe(false)
    expect(s.dry).toBe(false)
    expect(s.thinking).toBe(true)
    expect(s.thinkingOff).toBe(true)
  })

  it('cannot switch thinking off when the model always thinks', () => {
    const s = supportsOf({ cloud: { info: info({ reasoning: { mandatory: true, efforts: [], supportsMaxTokens: false } }) } })!
    expect(s.thinking).toBe(true)
    expect(s.thinkingOff).toBe(false)
    expect(s.thinkingBudget).toBe(false)
    const none = supportsOf({ cloud: { info: info({ reasoning: null }) } })!
    expect(none.thinking).toBe(false)
  })
})

describe('is it a cloud model', () => {
  it('trusts the entry when the list has it, else the id', () => {
    const models = [{ id: 'openrouter/acme/big', cloud: {} }, { id: 'openrouter/folder/file.gguf' }]
    expect(isCloudId('openrouter/acme/big', models)).toBe(true)
    expect(isCloudId('openrouter/folder/file.gguf', models)).toBe(false)
    expect(isCloudId('openrouter/unknown/model', models)).toBe(true)
    expect(isCloudId('Qwen/q.gguf', models)).toBe(false)
    expect(isCloudId(null, models)).toBe(false)
  })
})

describe('where a batch run on a cloud model runs', () => {
  const id = cloudModelId('acme/big')

  it('has exactly one choice: OpenRouter', () => {
    const c = hostChoices(id, [], true, true)
    expect(c).toHaveLength(1)
    expect(c[0].host).toEqual({ kind: 'cloud' })
    expect(c[0].label).toBe('OpenRouter')
    expect(c[0].valid).toBe(true)
    expect(hostKey(c[0].host)).toBe('cloud')
  })

  it('needs a stdio REA like a rental does', () => {
    const c = hostChoices(id, [], false, true)
    expect(c[0].valid).toBe(false)
    expect(c[0].why).toMatch(/stdio/)
  })

  it('does not change the choices of other models', () => {
    const c = hostChoices('Qwen/q.gguf', [], true)
    expect(c.map((x) => x.host.kind)).toEqual(['local'])
    expect(c[0].valid).toBe(true)
  })

  it('allows up to four tasks at once', () => {
    expect(maxSessions(1, true)).toBe(MAX_CLOUD_SESSIONS)
    expect(maxSessions(1)).toBe(8)
    const t = targetFor([{ kind: 'cloud' }], { sessionsPerHost: 9 })
    expect(t.sessionsPerHost).toBe(MAX_CLOUD_SESSIONS)
    expect(t.contextPerSession).toBeNull()
    expect(t.stopRentalsAfter).toBe(false)
    expect(targetFor([{ kind: 'cloud' }]).sessionsPerHost).toBe(1)
  })

  it('needs no context per session and refuses more than four tasks', () => {
    const choices = hostChoices(id, [], true, true)
    const ok = targetFor([{ kind: 'cloud' }], { sessionsPerHost: 3 })
    expect(targetProblems(ok, choices)).toEqual([])
    expect(targetProblems({ ...ok, sessionsPerHost: 6 }, choices)).toEqual([`At most ${MAX_CLOUD_SESSIONS} tasks at once on each.`])
  })
})

describe('the active model', () => {
  const inst = (modelId: string, startedAt: number, cloud = false): LoadedInstance => ({ instanceId: modelId, modelId, modelName: modelId, status: 'ready', startedAt, ...(cloud ? { cloud: { provider: 'openrouter', free: false } } : {}) }) as unknown as LoadedInstance

  it('never moves to an OpenRouter model by itself', () => {
    expect(pickActive([inst('openrouter/a', 9, true), inst('local', 1)], 'gone', null)).toBe('local')
    expect(pickActive([inst('openrouter/a', 9, true)], 'gone', null)).toBeNull()
  })

  it('keeps one the user connected', () => {
    expect(pickActive([inst('openrouter/a', 9, true), inst('local', 1)], 'openrouter/a', null)).toBe('openrouter/a')
    expect(pickActive([], 'openrouter/a', 'openrouter/a')).toBe('openrouter/a')
  })
})

describe('the cloud store', () => {
  it('keys a call by channel and arguments', () => {
    expect(callKey('cloud:addModel', ['a/b'])).not.toBe(callKey('cloud:addModel', ['a/c']))
    expect(callKey('cloud:checkKey', [])).toBe(callKey('cloud:checkKey', []))
  })

  it('never lets an older state replace a newer one', () => {
    const a = state({ rev: 5 })
    const b = state({ rev: 4 })
    expect(newer(a, b).state).toBe(a)
    expect(newer(b, a).state).toBe(a)
    expect(newer(null, b).state).toBe(b)
  })
})
