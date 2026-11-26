// The Coding mode, SettingsStore.setModeLoad, the chat request split, the shared practice helpers and the overview per
// model (docs/app-modes-design.md, section 1).
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { DEFAULT_LOAD_CONFIG, DEFAULT_SAMPLING } from '@shared/defaults'
import { APP_MODES, builtInSampling, MODE_LABELS, modeDefaultsEmpty, modeLoad, modePromptExtra, modeSampling, reloadFor, type ModeDefaults } from '@shared/modes'
import * as practice from '@shared/practice'
import type { BatchRun } from '@shared/batch'
import type { LoadConfig, ReSession, ReTraceInfo } from '@shared/types'
import { buildRequestBody, samplingRequestFields } from '../src/main/chat/service'
import { ModelGate } from '../src/main/modelGate'
import { buildOverview } from '../src/main/re/overview'
import { parseModeDefaults } from '../src/main/settings/schema'
import { SettingsStore } from '../src/main/settings/store'
import { scopeFromValue } from '../src/renderer/src/lib/modes'
import * as reTraces from '../src/renderer/src/stores/reTraces'

const dirs: string[] = []
const tmp = (): string => {
  const d = mkdtempSync(join(tmpdir(), 'rulz-coding-'))
  dirs.push(d)
  return d
}
afterEach(() => {
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true })
})
const load = (over: Partial<LoadConfig> = {}): LoadConfig => ({ ...DEFAULT_LOAD_CONFIG, ...over })
const entry = (over: Partial<ModeDefaults> = {}): ModeDefaults => ({ load: null, sampling: null, systemPrompt: '', ...over })

describe('the Coding mode', () => {
  it('is the last mode, with a label, and starts from the app\'s default sampling', () => {
    expect(APP_MODES.at(-1)).toBe('coding')
    expect(MODE_LABELS.coding).toBe('Coding')
    const app = { ...DEFAULT_SAMPLING, temperature: 0.33 }
    expect(builtInSampling('coding', app)).toBe(app)
    expect(modeSampling('coding', undefined, app)).toEqual(app)
    expect(modeSampling('coding', { coding: entry({ sampling: { ...DEFAULT_SAMPLING, topK: 9, jsonSchema: '{}' } }) }, app)).toMatchObject({ topK: 9, jsonSchema: '' })
  })

  it('has load settings, a prompt text and a reload offer like the others', () => {
    const base = load()
    const own = load({ contextLength: 65536 })
    expect(modeLoad(base, { coding: entry({ load: own }) }, 'coding')).toEqual(own)
    expect(modePromptExtra({ coding: entry({ systemPrompt: ' tools ' }) }, 'coding')).toBe('tools')
    expect(reloadFor(base, base, { coding: entry({ load: own }) }, 'coding')).toEqual(own)
    expect(modeDefaultsEmpty(entry({ systemPrompt: 'x' }))).toBe(false)
  })

  it('is kept in model-mode-defaults.json and survives a restart; an older unknown mode is still dropped', async () => {
    const dir = tmp()
    const store = new SettingsStore(dir)
    await store.init()
    await store.setModeDefaults('m', 'coding', entry({ load: load({ contextLength: 65536 }), sampling: { ...DEFAULT_SAMPLING, temperature: 0.5 }, systemPrompt: 'Use tools.' }))
    const again = new SettingsStore(dir)
    await again.init()
    expect(again.modeDefaults('m').coding).toMatchObject({ load: { contextLength: 65536 }, sampling: { temperature: 0.5, jsonSchema: '' }, systemPrompt: 'Use tools.' })
    expect(again.loadFor('m', 'coding').contextLength).toBe(65536)
    expect(again.loadFor('m', 're').contextLength).toBe(8192)
    expect(Object.keys(parseModeDefaults({ m: { coding: entry({ systemPrompt: 'x' }), nope: entry({ systemPrompt: 'y' }) } }).m)).toEqual(['coding'])
    expect(readFileSync(join(dir, 'model-mode-defaults.json'), 'utf8')).toContain('"coding"')
  })

  it('rejects a mode that does not exist when saving, and the renderer reads coding as a scope', async () => {
    const store = new SettingsStore(tmp())
    await store.init()
    await expect(store.setModeDefaults('m', 'nope' as never, entry({ systemPrompt: 'x' }))).rejects.toThrow(/Unknown mode/)
    expect(scopeFromValue('coding')).toBe('coding')
    expect(scopeFromValue('all')).toBeNull()
    expect(scopeFromValue('nope')).toBeNull()
  })
})

describe('SettingsStore.setModeLoad', () => {
  const open = async (): Promise<SettingsStore> => {
    const store = new SettingsStore(tmp())
    await store.init()
    return store
  }

  it('changes only that mode\'s load and keeps its sampling and prompt', async () => {
    const store = await open()
    await store.setModeDefaults('m', 'coding', entry({ sampling: { ...DEFAULT_SAMPLING, topK: 7 }, systemPrompt: 'hello' }))
    await store.setModeDefaults('m', 're', entry({ load: load({ contextLength: 1 + 4096 }) }))
    await store.setModeLoad('m', 'coding', load({ contextLength: 65536 }))
    expect(store.modeDefaults('m').coding).toMatchObject({ load: { contextLength: 65536 }, sampling: { topK: 7 }, systemPrompt: 'hello' })
    expect(store.modeDefaults('m').re?.load?.contextLength).toBe(4097)
  })

  it('gives a new load the model\'s API-server sampling and keeps what an entry already has', async () => {
    const store = await open()
    const sampling = { ...DEFAULT_LOAD_CONFIG.sampling, temperature: 1, override: true }
    await store.setModelDefaults('m', load({ sampling }))
    await store.setModeLoad('m', 'coding', load({ contextLength: 4096, sampling: { ...DEFAULT_LOAD_CONFIG.sampling, temperature: 3 } }))
    expect(store.modeDefaults('m').coding?.load?.sampling).toEqual(sampling)
    await store.setModelDefaults('m', load())
    await store.setModeLoad('m', 'coding', load({ contextLength: 2048 }))
    expect(store.modeDefaults('m').coding?.load?.sampling).toEqual(sampling)
  })

  it('removes the load with null, and the entry (and the model) when nothing is left', async () => {
    const store = await open()
    await store.setModeLoad('m', 'coding', load({ contextLength: 4096 }))
    await store.setModeDefaults('m', 're', entry({ systemPrompt: 'keep' }))
    await store.setModeLoad('m', 'coding', null)
    expect(store.modeDefaults('m')).toEqual({ re: entry({ systemPrompt: 'keep' }) })
    await store.setModeLoad('m', 're', null)
    expect(store.modeDefaults('m').re).toMatchObject({ load: null, systemPrompt: 'keep' })
    await store.setModeDefaults('m', 're', null)
    expect(store.allModeDefaults()).toEqual({})
    await store.setModeLoad('other', 'coding', null)
    expect(store.allModeDefaults()).toEqual({})
  })

  it('goes through the same queue as other saves, so none is lost, and refuses unknown modes', async () => {
    const store = await open()
    await Promise.all([
      store.setModeDefaults('m', 'coding', entry({ sampling: { ...DEFAULT_SAMPLING, topK: 3 }, systemPrompt: 'p' })),
      store.setModeLoad('m', 'coding', load({ contextLength: 4096 })),
      store.setModeDefaults('m', 're', entry({ systemPrompt: 'r' })),
      store.setModeLoad('m', 're', load({ contextLength: 2048 }))
    ])
    const m = store.modeDefaults('m')
    expect(m.coding).toMatchObject({ load: { contextLength: 4096 }, sampling: { topK: 3 }, systemPrompt: 'p' })
    expect(m.re).toMatchObject({ load: { contextLength: 2048 }, systemPrompt: 'r' })
    await expect(store.setModeLoad('m', 'nope' as never, load())).rejects.toThrow(/Unknown mode/)
  })
})

describe('the chat request and the sampling fields of a mode', () => {
  const busy = {
    ...DEFAULT_SAMPLING,
    maxTokens: 100,
    stop: ['###'],
    seed: 7,
    dryMultiplier: 0.8,
    dryPenaltyLastN: 64,
    xtcProbability: 0.3,
    avoidWords: ['moist'],
    thinking: 'on' as const,
    reasoningEffort: 'high' as const,
    thinkingBudget: 512,
    thinkingBudgetMessage: 'wrap up',
    jsonSchema: '{"type":"object"}'
  }

  it('builds the request with the same keys, in the same order, as before the split', () => {
    const body = buildRequestBody('m', [], busy, false, [{ id: 0, scale: 0.5 }])
    expect(Object.keys(body)).toEqual([
      'model',
      'messages',
      'stream',
      'return_progress',
      'temperature',
      'top_k',
      'top_p',
      'min_p',
      'repeat_penalty',
      'presence_penalty',
      'frequency_penalty',
      'lora',
      'max_tokens',
      'stop',
      'seed',
      'dry_multiplier',
      'dry_base',
      'dry_allowed_length',
      'dry_penalty_last_n',
      'dry_sequence_breakers',
      'xtc_probability',
      'xtc_threshold',
      'logit_bias',
      'chat_template_kwargs',
      'reasoning_effort',
      'thinking_budget_tokens',
      'reasoning_budget_message',
      'response_format'
    ])
    expect(Object.keys(buildRequestBody('m', [], DEFAULT_SAMPLING))).toEqual(['model', 'messages', 'stream', 'return_progress', 'temperature', 'top_k', 'top_p', 'min_p', 'repeat_penalty', 'presence_penalty', 'frequency_penalty'])
  })

  it('gives the sampling and thinking parts the chat request is built from, without what belongs to one request', () => {
    const body = buildRequestBody('m', [], busy)
    const { sampling, thinking } = samplingRequestFields(busy)
    for (const [k, v] of Object.entries({ ...sampling, ...thinking })) expect(body[k]).toEqual(v)
    for (const k of ['model', 'messages', 'stream', 'return_progress', 'max_tokens', 'stop', 'seed', 'response_format', 'lora']) {
      expect(sampling).not.toHaveProperty(k)
      expect(thinking).not.toHaveProperty(k)
    }
    expect(sampling).toMatchObject({ temperature: DEFAULT_SAMPLING.temperature, dry_multiplier: 0.8, dry_penalty_last_n: 64, xtc_probability: 0.3 })
    expect(thinking).toEqual({ chat_template_kwargs: { enable_thinking: true }, reasoning_effort: 'high', thinking_budget_tokens: 512, reasoning_budget_message: 'wrap up' })
    // Thinking off sends neither effort nor budget; auto sends nothing at all.
    expect(samplingRequestFields({ ...busy, thinking: 'off' }).thinking).toEqual({ chat_template_kwargs: { enable_thinking: false } })
    expect(samplingRequestFields(DEFAULT_SAMPLING).thinking).toEqual({})
    expect(samplingRequestFields(busy, true).thinking).toEqual({ chat_template_kwargs: { enable_thinking: false } })
  })
})

describe('the model gate for a reload', () => {
  it('says the app is loading a model, and holds the gate until it is released', () => {
    const gate = new ModelGate()
    const release = gate.take('reload:m') as () => void
    expect(gate.busyReason()).toBe('LLM Suite is loading a model.')
    expect(gate.take('adventure:1')).toBe('LLM Suite is loading a model.')
    release()
    expect(gate.busyReason()).toBeNull()
    expect(typeof gate.take('adventure:1')).toBe('function')
  })
})

describe('the practice helpers in src/shared/practice.ts', () => {
  it('are the ones the renderer store re-exports', () => {
    for (const k of ['modelKey', 'keyOfModel', 'practiceRecords', 'practiceOf', 'taskRecords', 'solvedTasks'] as const) expect(reTraces[k]).toBe(practice[k])
    expect(practice.modelKey('C:\\LLM\\Orn-1.5-Q4_K_M-00001-of-00002.gguf')).toBe('orn-1.5-q4_k_m')
  })
})

describe('buildOverview for one model', () => {
  const re = (extra: Partial<ReSession> = {}): ReSession => ({ target: 'C:\\a.exe', goal: 'g', collect: true, outcome: 'unrated', ...extra })
  const prac = (taskId: string) => ({ taskId, family: 'f', tier: 't1' })
  const task = (id: string) => ({ id, family: 'f', tier: 't1', binary: `${id}.exe`, prompt: 'p', set: 'original' })
  const apiTrace = (id: string, taskId: string, outcome: 'solved' | 'failed', models: string[]): ReTraceInfo => ({
    conversationId: id,
    responses: 1,
    toolRounds: 1,
    trimmed: false,
    models,
    bytes: 1,
    updatedAt: 5,
    api: { client: 'RulZ Coding Agent', title: id, target: 't', goal: 'g', outcome, createdAt: 1, kind: 'practice', practice: prac(taskId) }
  })
  const library = [
    { id: 'A/orn.gguf', name: 'Orn' },
    { id: 'B/other.gguf', name: 'Other' }
  ]
  const input = () => ({
    chats: [
      { id: 'c1', title: 'one', updatedAt: 10, modelId: 'A/orn.gguf', re: re({ outcome: 'solved', practice: { ...prac('t-1'), answer: '1', expected: '1' } }) },
      { id: 'c2', title: 'two', updatedAt: 11, modelId: 'B/other.gguf', re: re({ outcome: 'failed', practice: { ...prac('t-1'), answer: '2', expected: '1' } }) }
    ],
    traces: [apiTrace('api-1', 't-2', 'failed', ['Orn']), apiTrace('api-2', 't-2', 'solved', ['Other'])],
    practice: { kitDir: 'k', tasks: [task('t-1'), task('t-2'), task('t-3')], sets: [] },
    recording: true,
    summaries: [
      { id: 'c1', rePracticeTask: 't-1', reOutcome: 'solved' as const, modelId: 'A/orn.gguf' },
      { id: 'c2', rePracticeTask: 't-1', reOutcome: 'failed' as const, modelId: 'B/other.gguf' }
    ],
    runs: [
      {
        modelId: 'A/orn.gguf',
        modelName: 'Orn',
        // A scored task whose chat was deleted since still counts, for the run's model.
        items: [{ taskId: 't-3', status: 'wrong', conversationId: 'gone' }, { taskId: 't-1', status: 'solved', conversationId: 'c1' }]
      }
    ] as unknown as Pick<BatchRun, 'modelId' | 'modelName' | 'items'>[],
    models: library
  })

  it('without a model, counts the listed sessions as before and only adds the models of each session', () => {
    const o = buildOverview(input())
    expect(o.practice).not.toHaveProperty('model')
    expect(o.practice.tasks.map((t) => [t.id, t.attempts, t.solved])).toEqual([['t-1', 2, 1], ['t-2', 2, 1], ['t-3', 0, 0]])
    expect(o.practice.tasks[0]).not.toHaveProperty('failed')
    expect(o.sessions.map((s) => [s.id, s.models])).toEqual([
      ['c2', ['other']],
      ['c1', ['orn']],
      ['api-1', ['orn']],
      ['api-2', ['other']]
    ])
  })

  it('counts only that model\'s attempts, solved and failed, from the same records as the Reverse Engineering page', () => {
    const o = buildOverview({ ...input(), model: 'A/orn.gguf' })
    expect(o.practice.model).toBe('orn')
    expect(o.practice.tasks.map((t) => [t.id, t.attempts, t.solved, t.failed])).toEqual([
      ['t-1', 1, 1, 0],
      ['t-2', 1, 0, 1],
      // from the batch run whose chat was deleted
      ['t-3', 1, 0, 1]
    ])
    const other = buildOverview({ ...input(), model: 'B/other.gguf' })
    expect(other.practice.tasks.map((t) => [t.id, t.attempts, t.solved, t.failed])).toEqual([
      ['t-1', 1, 0, 1],
      ['t-2', 1, 1, 0],
      ['t-3', 0, 0, 0]
    ])
  })

  it('matches a model that is not in the library by the text the client sent', () => {
    const o = buildOverview({ ...input(), model: 'Orn' })
    expect(o.practice.model).toBe('orn')
    expect(o.practice.tasks.find((t) => t.id === 't-1')).toMatchObject({ attempts: 1, solved: 1 })
    const none = buildOverview({ ...input(), model: 'Never-Heard-Of' })
    expect(none.practice.model).toBe('never-heard-of')
    expect(none.practice.tasks.every((t) => t.attempts === 0 && t.failed === 0)).toBe(true)
  })
})
