// Modes and rentals (RulZ LLM Suite 0.35): reading its answers, merging sampling, choosing a model,
// the client's new routes and the rules the window and the phone share.
import { describe, expect, it, vi } from 'vitest'
import { familyCounts } from '@shared/batch'
import { DEFAULT_SETTINGS } from '@shared/defaults'
import {
  MAX_MODE_PROMPT_CHARS,
  chooseModel,
  cleanRequestFields,
  describeModeUse,
  modeHeaders,
  modePrompt,
  modesStamp,
  parseModelModes,
  pickDefaultModel,
  practiceScope,
  reloadOffer,
  rentalGoneMessage,
  rentalOf,
  requestSkipsThinking,
  taskSamplingFields,
  triesText,
  withApiOverride
} from '@shared/modes'
import type { ModelInfo, ModelModes } from '@shared/types'
import { friendlyServerError, parseModels, RulzClient, ServerError } from '../src/main/rulz/client'
import { modelAdmin, RENTAL_SETTINGS_REFUSAL } from '../src/main/rulz/modelAdmin'
import { startMockRulz } from './helpers/mockRulz'

const QWEN = 'Qwen3-Coder-30B-A3B-Instruct-Q5_K_M'

const modes = (patch: { coding?: Partial<ModelModes['modes']['coding']>; re?: Partial<ModelModes['modes']['re']>; apiOverride?: Record<string, unknown> | null } = {}): ModelModes => ({
  id: QWEN,
  state: 'ready',
  instance: 'i1',
  apiOverride: patch.apiOverride ?? null,
  modes: {
    coding: { ownLoad: false, sampling: null, thinking: null, systemPrompt: '', matches: true, ...patch.coding },
    re: { ownLoad: false, sampling: null, thinking: null, systemPrompt: '', matches: true, ...patch.re }
  }
})

describe('reading the modes body (untrusted)', () => {
  const body = {
    object: 'model.modes',
    id: QWEN,
    state: 'ready',
    instance: 'abc',
    api_override: { temperature: 1 },
    modes: {
      coding: {
        own_load: true,
        sampling: { temperature: 0.6, top_k: 20 },
        thinking: { chat_template_kwargs: { enable_thinking: true } },
        system_prompt: 'Be brief.',
        matches: false
      },
      re: { own_load: false, sampling: null, thinking: null, system_prompt: '', matches: null }
    }
  }

  it('reads a good body', () => {
    expect(parseModelModes(body)).toEqual({
      id: QWEN,
      state: 'ready',
      instance: 'abc',
      apiOverride: { temperature: 1 },
      modes: {
        coding: { ownLoad: true, sampling: { temperature: 0.6, top_k: 20 }, thinking: { chat_template_kwargs: { enable_thinking: true } }, systemPrompt: 'Be brief.', matches: false },
        re: { ownLoad: false, sampling: null, thinking: null, systemPrompt: '', matches: null }
      }
    })
  })

  it('throws when there is no modes object, and fills in what is missing or has the wrong type', () => {
    expect(() => parseModelModes(null)).toThrow(/could not read/)
    expect(() => parseModelModes({ id: 'x' })).toThrow(/could not read/)
    expect(() => parseModelModes('modes')).toThrow(/could not read/)
    const odd = parseModelModes({ id: 7, state: 'exploding', instance: 5, api_override: 'x', modes: { coding: { own_load: 'yes', sampling: 'hot', system_prompt: 9, matches: 'no' }, re: 3 } })
    expect(odd).toEqual({
      id: '',
      state: 'not-loaded',
      instance: null,
      apiOverride: null,
      modes: {
        coding: { ownLoad: false, sampling: null, thinking: null, systemPrompt: '', matches: null },
        re: { ownLoad: false, sampling: null, thinking: null, systemPrompt: '', matches: null }
      }
    })
  })

  it('cuts the prompt at 20,000 characters', () => {
    const long = parseModelModes({ modes: { coding: { system_prompt: 'x'.repeat(30_000) } } })
    expect(long.modes.coding.systemPrompt).toHaveLength(MAX_MODE_PROMPT_CHARS)
    expect(MAX_MODE_PROMPT_CHARS).toBe(20_000)
  })

  it('strips fields that must stay this app\'s, and anything that is not plain data', () => {
    const cleaned = cleanRequestFields({
      temperature: 0.5,
      model: 'evil',
      messages: [{ role: 'user', content: 'x' }],
      tools: [],
      stream: false,
      max_tokens: 1,
      max_completion_tokens: 1,
      response_format: { type: 'json_object' },
      seed: 3,
      stop: ['a'],
      ['__proto__']: { polluted: true },
      'Bad Name': 1,
      nan: Number.NaN,
      huge: 'x'.repeat(5000),
      fn: () => 1,
      deep: { a: { b: { c: 1 } } },
      top_k: 40
    })
    expect(Object.keys(cleaned ?? {}).sort()).toEqual(['deep', 'temperature', 'top_k'])
    expect(({} as Record<string, unknown>).polluted).toBeUndefined()
    expect(cleanRequestFields('x')).toBeNull()
    expect(cleanRequestFields([1])).toBeNull()
    expect(cleanRequestFields(null)).toBeNull()
  })

  it('never lets a forbidden key through parseModelModes', () => {
    const m = parseModelModes({ modes: { coding: { sampling: { temperature: 0.5, model: 'x', messages: [], max_tokens: 5 }, thinking: { stream: true, reasoning_effort: 'high' } } } })
    expect(m.modes.coding.sampling).toEqual({ temperature: 0.5 })
    expect(m.modes.coding.thinking).toEqual({ reasoning_effort: 'high' })
  })
})

describe('which sampling a task uses', () => {
  const suite = modes({
    coding: {
      sampling: { temperature: 0.6, top_k: 40 },
      thinking: { chat_template_kwargs: { enable_thinking: true }, reasoning_effort: 'high' }
    }
  })

  it('Coding: recommended, then LLM Suite, then the values set here', () => {
    expect(taskSamplingFields(QWEN, { temperature: 0.9 }, 'coding', suite)).toEqual({
      temperature: 0.9,
      top_p: 0.8,
      top_k: 40,
      repeat_penalty: 1.05,
      chat_template_kwargs: { enable_thinking: true },
      reasoning_effort: 'high'
    })
  })

  it('Coding: thinking turned off here removes LLM Suite\'s thinking fields', () => {
    const off = taskSamplingFields(QWEN, { thinking: 'off' }, 'coding', suite)
    expect(off.chat_template_kwargs).toEqual({ enable_thinking: false })
    expect(off).not.toHaveProperty('reasoning_effort')
    const budget0 = taskSamplingFields(QWEN, { thinkingBudget: 0 }, 'coding', suite)
    expect(budget0).not.toHaveProperty('chat_template_kwargs')
    expect(budget0).not.toHaveProperty('reasoning_effort')
    expect(budget0.thinking_budget_tokens).toBe(0)
    expect(requestSkipsThinking(off)).toBe(true)
    expect(requestSkipsThinking(budget0)).toBe(true)
    expect(requestSkipsThinking(taskSamplingFields(QWEN, undefined, 'coding', suite))).toBe(false)
  })

  it('Coding without sampling in LLM Suite, or an older LLM Suite: as before', () => {
    expect(taskSamplingFields(QWEN, { temperature: 0.2 }, 'coding', null)).toEqual({ temperature: 0.2, top_p: 0.8, top_k: 20, repeat_penalty: 1.05 })
    expect(taskSamplingFields(QWEN, { temperature: 0.2 }, 'coding', modes())).toEqual({ temperature: 0.2, top_p: 0.8, top_k: 20, repeat_penalty: 1.05 })
    expect(taskSamplingFields('gemma', undefined, 'coding', null)).toEqual({})
  })

  it('Reverse engineering: LLM Suite\'s own sampling as it is, otherwise as before', () => {
    const re = modes({ re: { sampling: { temperature: 0.1 }, thinking: { chat_template_kwargs: { enable_thinking: false } } } })
    expect(taskSamplingFields(QWEN, { temperature: 0.9, topK: 3 }, 're', re)).toEqual({ temperature: 0.1, chat_template_kwargs: { enable_thinking: false } })
    expect(taskSamplingFields(QWEN, { temperature: 0.9 }, 're', modes())).toEqual({ temperature: 0.9, top_p: 0.8, top_k: 20, repeat_penalty: 1.05 })
    expect(taskSamplingFields(QWEN, { temperature: 0.9 }, 're', null)).toEqual({ temperature: 0.9, top_p: 0.8, top_k: 20, repeat_penalty: 1.05 })
  })

  it('the API server override is what really applies, for the thinking status too', () => {
    const api = modes({ apiOverride: { chat_template_kwargs: { enable_thinking: false } } })
    const fields = taskSamplingFields(QWEN, undefined, 'coding', api)
    expect(requestSkipsThinking(fields)).toBe(false)
    expect(requestSkipsThinking(withApiOverride(fields, api))).toBe(true)
    expect(withApiOverride(fields, null)).toEqual(fields)
  })
})

describe('the line a task says about its settings', () => {
  it('says nothing when nothing of LLM Suite applies', () => {
    expect(describeModeUse('coding', null, undefined)).toBeNull()
    expect(describeModeUse('coding', modes(), { temperature: 0.2 })).toBeNull()
    expect(describeModeUse('re', modes({ coding: { sampling: { temperature: 1 } } }), undefined)).toBeNull()
  })

  it('names what is used and what is overridden', () => {
    const m = modes({ re: { sampling: { temperature: 0.1 }, systemPrompt: 'Go.' }, coding: { sampling: { temperature: 0.6, top_k: 20 } } })
    expect(describeModeUse('re', m, undefined)).toBe("Uses LLM Suite's Reverse engineering sampling and prompt.")
    expect(describeModeUse('coding', m, undefined)).toBe("Uses LLM Suite's Coding sampling.")
    expect(describeModeUse('coding', m, { temperature: 0.9, topK: 5, topP: 0.5 })).toBe("Uses LLM Suite's Coding sampling. 2 values set here override it.")
    expect(describeModeUse('coding', m, { temperature: 0.9 })).toBe("Uses LLM Suite's Coding sampling. 1 value set here overrides it.")
    expect(describeModeUse('coding', modes({ coding: { systemPrompt: 'x' } }), undefined)).toBe("Uses LLM Suite's Coding prompt.")
  })

  it('names the API server override on its own', () => {
    expect(describeModeUse('coding', modes({ apiOverride: { temperature: 1 } }), undefined)).toBe('API server sampling overrides temperature.')
    expect(describeModeUse('coding', modes({ apiOverride: { top_k: 1, temperature: 1 } }), undefined)).toBe('API server sampling overrides top k, temperature.')
  })

  it('the prompt is only there when LLM Suite has one', () => {
    expect(modePrompt('coding', modes())).toBeNull()
    expect(modePrompt('coding', null)).toBeNull()
    expect(modePrompt('re', modes({ re: { systemPrompt: '  Be careful.  ' } }))).toEqual({ label: 'Reverse engineering', text: 'Be careful.' })
  })
})

describe('headers', () => {
  it('always carry the mode, and deny rentals unless allowed', () => {
    expect(modeHeaders('coding', false)).toEqual({ 'X-RulZ-Mode': 'coding', 'X-RulZ-Rental': 'deny' })
    expect(modeHeaders('re', true)).toEqual({ 'X-RulZ-Mode': 're' })
  })
})

describe('rentals in the model list', () => {
  it('reads `rental`, or a vast/ path from LLM Suite 0.29 to 0.34', () => {
    expect(rentalOf({ rental: { label: 'RTX 5090', reconnecting: true } })).toEqual({ label: 'RTX 5090', reconnecting: true })
    expect(rentalOf({ rental: {} })).toEqual({ label: null, reconnecting: false })
    expect(rentalOf({ path: 'vast/12345/model.gguf' })).toEqual({ label: null, reconnecting: false })
    expect(rentalOf({ path: 'C:\\models\\a.gguf' })).toBeUndefined()
    expect(rentalOf({})).toBeUndefined()
    const { models } = parseModels({
      data: [
        { id: 'a', owned_by: 'rulz', path: 'C:\\m\\a.gguf', state: 'loaded' },
        { id: 'b', owned_by: 'rulz', path: 'vast/1/b.gguf', state: 'loaded' },
        { id: 'c', owned_by: 'rulz', path: 'x', state: 'loaded', rental: { label: 'L', reconnecting: false } }
      ]
    })
    expect(models.map((m) => !!m.rental)).toEqual([false, true, true])
  })

  const local = (id: string, loaded = false): ModelInfo => ({ id, loaded })
  const rented = (id: string): ModelInfo => ({ id, loaded: true, rental: { label: 'R', reconnecting: false } })

  it('never picks a rental\'s model by itself', () => {
    expect(pickDefaultModel([rented('Qwen3-Coder-big'), local('gemma')])).toBe('gemma')
    expect(pickDefaultModel([rented('Qwen3-Coder-big')])).toBe('')
    expect(chooseModel('', false, [rented('Qwen3-Coder-big'), local('gemma')])).toEqual({ ok: true, model: 'gemma', rental: false })
    // A local model that is gone is swapped as today, still never for a rental's.
    expect(chooseModel('old', false, [rented('r'), local('gemma')])).toEqual({ ok: true, model: 'gemma', rental: false })
    const only = chooseModel('', false, [rented('r')])
    expect(only.ok).toBe(false)
    expect(chooseModel('', false, [])).toMatchObject({ ok: false, reason: expect.stringContaining('no models yet') })
  })

  it('uses a rental\'s model when it was chosen and is listed, and says so when it is gone', () => {
    expect(chooseModel('r', true, [rented('r'), local('gemma')])).toEqual({ ok: true, model: 'r', rental: true })
    expect(chooseModel('gemma', false, [rented('r'), local('gemma')])).toEqual({ ok: true, model: 'gemma', rental: false })
    const gone = chooseModel('org/Big-Model.gguf', true, [local('gemma')])
    expect(gone).toEqual({ ok: false, reason: "Big-Model is not loaded on a rental now. Load it on LLM Suite's vast.ai page, or choose another model." })
    expect(rentalGoneMessage('x/y.gguf')).toBe("y is not loaded on a rental now. Load it on LLM Suite's vast.ai page, or choose another model.")
    // Without a list nothing is known: the chosen model is used, and counts as local (so requests deny rentals).
    expect(chooseModel('r', true, [])).toEqual({ ok: true, model: 'r', rental: false })
  })

  it('a model list entry that is a rental now wins over an old "was a rental" flag', () => {
    expect(chooseModel('gemma', true, [local('gemma')])).toEqual({ ok: true, model: 'gemma', rental: false })
    expect(chooseModel('gemma', false, [rented('gemma')])).toEqual({ ok: true, model: 'gemma', rental: true })
  })
})

describe('the reload bar', () => {
  it('shows when the running settings differ from the mode\'s, until "Keep as is" for that instance', () => {
    const stale = modes({ coding: { matches: false } })
    expect(reloadOffer(stale, 'coding', null)).toEqual({ label: 'Reload with Coding settings' })
    expect(reloadOffer(stale, 're', null)).toBeNull()
    expect(reloadOffer(modes({ re: { matches: false } }), 're', null)).toEqual({ label: 'Reload with Reverse engineering settings' })
    expect(reloadOffer(stale, 'coding', 'i1')).toBeNull()
    expect(reloadOffer(stale, 'coding', 'i0')).not.toBeNull()
    expect(reloadOffer(modes({ coding: { matches: null } }), 'coding', null)).toBeNull()
    expect(reloadOffer({ ...stale, state: 'not-loaded' }, 'coding', null)).toBeNull()
    expect(reloadOffer(null, 'coding', null)).toBeNull()
  })

  it('asks for the modes again when the loaded state, context or rental flag changes', () => {
    const a = modesStamp({ id: 'a', loaded: true, loadedContext: 8192 })
    expect(modesStamp({ id: 'a', loaded: true, loadedContext: 8192 })).toBe(a)
    expect(modesStamp({ id: 'a', loaded: true, loadedContext: 16384 })).not.toBe(a)
    expect(modesStamp({ id: 'a', loaded: false })).not.toBe(a)
    expect(modesStamp({ id: 'a', loaded: true, loadedContext: 8192, rental: { label: null, reconnecting: false } })).not.toBe(a)
    expect(modesStamp(undefined)).toBe('')
  })
})

describe('practice per model', () => {
  it('says which model the counts are for', () => {
    expect(practiceScope({ kitDir: 'k', model: 'qwen', tasks: [] })).toBe('For qwen')
    expect(practiceScope({ kitDir: 'k', tasks: [] })).toBe('All models together')
    expect(practiceScope({ kitDir: 'k', model: 'a/b.gguf', tasks: [] }, (s) => s.toUpperCase())).toBe('For A/B.GGUF')
  })

  it('shows failed counts where they are sent', () => {
    expect(triesText({ attempts: 3, solved: 1, failed: 2 })).toBe('solved 1/3 · 2 failed')
    expect(triesText({ attempts: 2, solved: 0, failed: 2 })).toBe('0/2 · 2 failed')
    expect(triesText({ attempts: 2, solved: 1 })).toBe('solved 1/2')
    const tile = familyCounts([
      { id: 'a', family: 'xor', tier: 't', binary: '', prompt: '', attempts: 2, solved: 1, failed: 1 },
      { id: 'b', family: 'xor', tier: 't', binary: '', prompt: '', attempts: 1, solved: 0, failed: 1 }
    ])
    expect(tile).toEqual([{ family: 'xor', total: 2, tried: 2, solved: 1, failed: 2 }])
    expect(familyCounts([{ id: 'a', family: 'xor', tier: 't', binary: '', prompt: '', attempts: 1, solved: 0 }])[0]).not.toHaveProperty('failed')
  })
})

describe('the client\'s routes for LLM Suite 0.35', () => {
  const body = {
    object: 'model.modes',
    id: 'org/Model name',
    state: 'ready',
    instance: 'i7',
    api_override: null,
    modes: {
      coding: { own_load: true, sampling: { temperature: 0.6 }, thinking: null, system_prompt: 'P', matches: false },
      re: { own_load: false, sampling: null, thinking: null, system_prompt: '', matches: true }
    }
  }

  it('reads the modes of a model whose id has slashes, with the headers it is given; 404 is an older LLM Suite', async () => {
    const newer = await startMockRulz({ modes: body })
    const older = await startMockRulz()
    try {
      const c = new RulzClient(() => ({ url: newer.url, apiKey: '' }))
      const got = await c.modelModes('org/Model name', undefined, modeHeaders('coding', false))
      expect(got?.modes.coding).toMatchObject({ sampling: { temperature: 0.6 }, systemPrompt: 'P', matches: false })
      expect(newer.headerLog).toEqual([{ path: '/v1/models/org%2FModel%20name/modes', method: 'GET', model: 'org/Model name', mode: 'coding', rental: 'deny' }])
      expect(await new RulzClient(() => ({ url: older.url, apiKey: '' })).modelModes('anything')).toBeNull()
    } finally {
      await newer.close()
      await older.close()
    }
  })

  it('an unreadable body or a server error is an error, not "older"', async () => {
    const bad = await startMockRulz({ modes: () => ({ status: 200, body: { nothing: true } }) })
    const down = await startMockRulz({ modes: () => ({ status: 500, body: { error: { message: 'boom' } } }) })
    try {
      await expect(new RulzClient(() => ({ url: bad.url, apiKey: '' })).modelModes('m')).rejects.toThrow(/could not read/)
      await expect(new RulzClient(() => ({ url: down.url, apiKey: '' })).modelModes('m')).rejects.toMatchObject({ message: 'boom', status: 500 })
    } finally {
      await bad.close()
      await down.close()
    }
  })

  it('reads and saves a mode\'s own load settings on the per-mode path, and removes them with null', async () => {
    const mock = await startMockRulz({ modes: body, loadConfig: true, reloadBlocked: 'A task is using the model.' })
    try {
      const c = new RulzClient(() => ({ url: mock.url, apiKey: '' }))
      const got = await c.loadSettings('org/m', 'coding')
      expect(got).toMatchObject({ mode: 'coding', modeOwn: false })
      const dry = await c.previewLoadSettings('org/m', { parallel: 2 }, 're')
      expect(dry.mode).toBe('re')
      expect(mock.modeLoadConfigPosts).toEqual([])
      const saved = await c.setLoadSettings('org/m', { parallel: 2 }, 'coding')
      expect(saved).toMatchObject({ mode: 'coding', modeOwn: true, reloadBlocked: 'A task is using the model.' })
      expect(mock.modeLoadConfigPosts).toEqual([{ mode: 'coding', model: 'org/m', body: { config: { parallel: 2 }, reload: true } }])
      await c.setLoadSettings('org/m', null, 'coding')
      expect(mock.modeLoadConfigPosts[1].body).toEqual({ config: null, reload: true })
      // The old path (all modes) is untouched by any of this.
      expect(mock.loadConfigPosts).toEqual([])
      await expect(c.setLoadSettings('org/m', null)).rejects.toThrow(/Pick a mode/)
    } finally {
      await mock.close()
    }
  })

  it('without per-mode load settings (older LLM Suite) the change says so instead of touching all modes', async () => {
    const older = await startMockRulz({ loadConfig: true })
    try {
      const c = new RulzClient(() => ({ url: older.url, apiKey: '' }))
      await expect(c.setLoadSettings('m', { parallel: 2 }, 'coding')).rejects.toThrow(/no load settings per mode/)
      expect(older.loadConfigPosts).toEqual([])
    } finally {
      await older.close()
    }
  })

  it('reloads with a mode, and shows LLM Suite\'s sentence when it refuses', async () => {
    const ok = await startMockRulz({ modes: body })
    const refused = await startMockRulz({ modes: body, reloadRefusal: 'The model is in use by a chat.' })
    const older = await startMockRulz()
    try {
      const done = await new RulzClient(() => ({ url: ok.url, apiKey: '' })).reloadMode('org/m', 're')
      expect(done.instance).toBe('i7')
      expect(ok.reloads).toEqual([{ model: 'org/m', body: { mode: 're' } }])
      const err = await new RulzClient(() => ({ url: refused.url, apiKey: '' })).reloadMode('m', 'coding').catch((e: unknown) => e)
      expect(err).toBeInstanceOf(ServerError)
      expect((err as ServerError).message).toBe('The model is in use by a chat.')
      expect(friendlyServerError(err)).toBe('The model is in use by a chat.')
      await expect(new RulzClient(() => ({ url: older.url, apiKey: '' })).reloadMode('m', 'coding')).rejects.toThrow(/cannot reload/)
    } finally {
      await ok.close()
      await refused.close()
      await older.close()
    }
  })

  it('asks the practice overview for one model, and an older LLM Suite counts all models together', async () => {
    const newer = await startMockRulz({ practicePerModel: true })
    const older = await startMockRulz()
    try {
      const n = new RulzClient(() => ({ url: newer.url, apiKey: '' }))
      const all = await n.reOverview()
      expect(all.practice.model).toBeUndefined()
      const per = await n.reOverview('org/Model a')
      expect(per.practice.model).toBe('org/Model a')
      expect(per.practice.tasks.find((t) => t.id === 'xor-1')).toMatchObject({ attempts: 2, solved: 1, failed: 1 })
      expect(newer.overviewQueries).toEqual(['', 'org/Model a'])
      const o = await new RulzClient(() => ({ url: older.url, apiKey: '' })).reOverview('org/Model a')
      expect(o.practice.model).toBeUndefined()
      expect(older.overviewQueries).toEqual(['org/Model a'])
    } finally {
      await newer.close()
      await older.close()
    }
  })

  it('shows a rental\'s own sentence for a refusal or a lost connection', () => {
    expect(friendlyServerError(new ServerError('x runs on a rented computer.', 409))).toBe('x runs on a rented computer.')
    expect(friendlyServerError(new ServerError('The connection to the rental "A" was lost. Reconnect it.', 502))).toBe('The connection to the rental "A" was lost. Reconnect it.')
    expect(friendlyServerError(new ServerError('boom', 500))).toContain('The model server reported a problem')
  })
})

describe('what the window and the phone may do to a model\'s settings', () => {
  const rental: ModelInfo = { id: 'rented', loaded: true, rental: { label: 'R', reconnecting: false } }
  const make = (idle = true) => {
    const client = {
      loadSettings: vi.fn(async () => ({ config: {} }) as never),
      previewLoadSettings: vi.fn(async () => ({ config: {} }) as never),
      setLoadSettings: vi.fn(async () => ({ config: {} }) as never),
      modelModes: vi.fn(async () => null),
      reloadMode: vi.fn(async () => modes())
    }
    const refreshed = vi.fn()
    const admin = modelAdmin({
      client,
      models: () => [{ id: 'local', loaded: true }, rental],
      assertIdle: (what) => {
        if (!idle) throw new Error(`A task is running. Wait for it to finish or press Stop, then ${what}.`)
      },
      refreshModels: refreshed
    })
    return { client, admin, refreshed }
  }

  it('reloads a mode and saves settings when nothing runs', async () => {
    const { client, admin, refreshed } = make()
    await admin.reloadMode('local', 'coding')
    expect(client.reloadMode).toHaveBeenCalledWith('local', 'coding')
    await admin.setLoadSettings('local', { parallel: 2 }, 're')
    expect(client.setLoadSettings).toHaveBeenCalledWith('local', { parallel: 2 }, 're')
    await admin.setLoadSettings('local', null, 're')
    expect(client.setLoadSettings).toHaveBeenLastCalledWith('local', null, 're')
    expect(refreshed).toHaveBeenCalledTimes(3)
  })

  it('refuses a reload and a save while a task or batch run goes', async () => {
    const { client, admin } = make(false)
    await expect(admin.reloadMode('local', 'coding')).rejects.toThrow(/reload the model/)
    await expect(admin.setLoadSettings('local', {}, 'coding')).rejects.toThrow(/change the load settings/)
    expect(client.reloadMode).not.toHaveBeenCalled()
    expect(client.setLoadSettings).not.toHaveBeenCalled()
  })

  it('refuses everything that would change or reload a rental\'s model, but lets its settings be read', async () => {
    const { client, admin } = make()
    await expect(admin.reloadMode('rented', 'coding')).rejects.toThrow(RENTAL_SETTINGS_REFUSAL)
    await expect(admin.setLoadSettings('rented', { parallel: 2 })).rejects.toThrow(RENTAL_SETTINGS_REFUSAL)
    await expect(admin.previewLoadSettings('rented', {})).rejects.toThrow(RENTAL_SETTINGS_REFUSAL)
    expect(client.reloadMode).not.toHaveBeenCalled()
    expect(client.setLoadSettings).not.toHaveBeenCalled()
    await admin.loadSettings('rented', 'coding')
    await admin.modes('rented')
    expect(client.loadSettings).toHaveBeenCalledWith('rented', 'coding')
  })
})

describe('settings', () => {
  it('start with no rental flag', () => {
    expect(DEFAULT_SETTINGS.modelRental).toBe(false)
  })
})
