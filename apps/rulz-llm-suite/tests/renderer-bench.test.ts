import { describe, expect, it } from 'vitest'
import type { BenchResult, BenchmarkRun } from '../src/shared/benchmark'
import type { LoadConfig } from '../src/shared/types'
import {
  availableAxes,
  axisValues,
  baseSummary,
  buildBenchRequest,
  defaultBenchForm,
  etaMs,
  fmtGbFromBytes,
  fmtMiB,
  fmtMs,
  fmtRatioPct,
  fmtSignedPct,
  fmtTps,
  insertRunIfAbsent,
  mergeRun,
  parseNumberList,
  runProgress,
  runsForModel,
  upsertRun,
  vramDiffPct
} from '../src/renderer/src/lib/bench'

const base = {
  contextLength: null,
  gpuLayers: 'auto',
  cpuMoe: 'off',
  cacheTypeK: 'f16',
  cacheTypeV: 'f16',
  flashAttention: 'auto',
  batchSize: null,
  ubatchSize: null,
  useDraft: false,
  draftTokens: null,
  adapters: [],
  sampling: {}
} as unknown as LoadConfig

const ctx = (available = availableAxes({ moe: false, hasDraft: false })) => ({ modelId: 'm.gguf', base, available })

describe('parseNumberList', () => {
  it('reads commas, spaces and a k suffix, and drops duplicates', () => {
    expect(parseNumberList('8192, 32k  8192;4096')).toEqual({ numbers: [8192, 32768, 4096], bad: [] })
  })
  it('reports entries that are not whole numbers in range', () => {
    const r = parseNumberList('10, abc, 1.5, -3, 0, 20', 1)
    expect(r.numbers).toEqual([10, 20])
    expect(r.bad).toEqual(['abc', '1.5', '-3', '0'])
  })
  it('accepts 0 when the minimum allows it', () => {
    expect(parseNumberList('0, 12', 0).numbers).toEqual([0, 12])
  })
  it('is empty for blank text', () => {
    expect(parseNumberList('  , ')).toEqual({ numbers: [], bad: [] })
  })
})

describe('axes of a model', () => {
  it('offers MoE offload only for expert models and draft only with a draft', () => {
    expect(availableAxes({ moe: false, hasDraft: false })).toEqual(['gpuLayers', 'cacheType', 'contextLength', 'batchSize', 'ubatchSize'])
    const all = availableAxes({ moe: true, hasDraft: true })
    expect(all).toContain('cpuMoe')
    expect(all).toContain('draft')
  })
  it('turns ticked choices and typed numbers into values in try order', () => {
    const f = defaultBenchForm()
    expect(axisValues('gpuLayers', { vary: true, picks: ['all', 'auto'], custom: '20' }).values).toEqual(['auto', 'all', 20])
    expect(axisValues('contextLength', f.axes.contextLength).values).toEqual([null, 8192, 32768])
    expect(axisValues('batchSize', f.axes.batchSize).values).toEqual([null, 512, 2048])
    expect(axisValues('draft', f.axes.draft).values).toEqual(['off', 2, 3])
    expect(axisValues('cacheType', f.axes.cacheType)).toEqual({ values: ['f16', 'q8_0'], bad: [] })
  })
})

describe('buildBenchRequest', () => {
  it('has no axes and no problem when nothing varies', () => {
    const b = buildBenchRequest(defaultBenchForm(), ctx())
    expect(b.request.axes).toEqual([])
    expect(b.problem).toBeNull()
    expect(b.request.prompt).toEqual({ promptTokens: 2048, genTokens: 128, repeats: 2 })
    expect(b.request.skipLikelyOom).toBe(true)
  })
  it('builds the varied axes and trims the label', () => {
    const form = defaultBenchForm()
    form.axes.gpuLayers.vary = true
    form.axes.cacheType.vary = true
    form.label = '  test  '
    const b = buildBenchRequest(form, ctx())
    expect(b.request.axes.map((a) => a.key)).toEqual(['gpuLayers', 'cacheType'])
    expect(b.request.label).toBe('test')
    expect(b.problem).toBeNull()
  })
  it('leaves out settings the model cannot vary', () => {
    const form = defaultBenchForm()
    form.axes.cpuMoe.vary = true
    form.axes.draft.vary = true
    expect(buildBenchRequest(form, ctx()).request.axes).toEqual([])
    const moe = buildBenchRequest(form, ctx(availableAxes({ moe: true, hasDraft: true })))
    expect(moe.request.axes.map((a) => a.key)).toEqual(['cpuMoe', 'draft'])
  })
  it('reports unreadable numbers by setting name', () => {
    const form = defaultBenchForm()
    form.axes.contextLength.vary = true
    form.axes.contextLength.custom = '8192, lots'
    expect(buildBenchRequest(form, ctx()).problem).toBe('Context: cannot use "lots".')
  })
  it('reports a varied setting with no values, too many combinations and bad prompts', () => {
    const form = defaultBenchForm()
    form.axes.cacheType.vary = true
    form.axes.cacheType.picks = []
    expect(buildBenchRequest(form, ctx()).problem).toContain('Pick at least one value for KV cache')

    const big = defaultBenchForm()
    big.axes.gpuLayers.vary = true
    big.axes.gpuLayers.custom = '10, 20, 30'
    big.axes.contextLength.vary = true
    big.axes.contextLength.custom = '1024, 2048, 4096'
    big.axes.batchSize.vary = true
    big.axes.batchSize.custom = '64, 128, 256'
    expect(buildBenchRequest(big, ctx()).problem).toContain('combinations')

    const bad = defaultBenchForm()
    bad.repeats = 0
    expect(buildBenchRequest(bad, ctx()).problem).toContain('Repeats')
  })
  it('summarizes the base settings', () => {
    expect(baseSummary(base)).toBe('GPU layers auto · context auto · KV cache f16 · flash attention auto · no draft · batch default/default')
  })
})

describe('formatting', () => {
  it('formats times, speeds and shares', () => {
    expect(fmtMs(850)).toBe('850 ms')
    expect(fmtMs(1250)).toBe('1.25 s')
    expect(fmtMs(125_000)).toBe('2m 5s')
    expect(fmtMs(undefined)).toBe('—')
    expect(fmtTps(41.234)).toBe('41.2')
    expect(fmtTps(undefined)).toBe('—')
    expect(fmtRatioPct(0.724)).toBe('72%')
    expect(fmtRatioPct(null)).toBe('—')
  })
  it('formats memory as GB', () => {
    expect(fmtMiB(2048)).toBe('2.0 GB')
    expect(fmtMiB(null)).toBe('—')
    expect(fmtGbFromBytes(3 * 1024 ** 3)).toBe('3.0 GB')
  })
  it('compares measured and estimated VRAM', () => {
    expect(vramDiffPct(1100, 1000 * 1024 * 1024)).toBeCloseTo(10)
    expect(vramDiffPct(900, 1000 * 1024 * 1024)).toBeCloseTo(-10)
    expect(vramDiffPct(null, 1000)).toBeNull()
    expect(vramDiffPct(1000, 0)).toBeNull()
    expect(fmtSignedPct(10.4)).toBe('+10%')
    expect(fmtSignedPct(-12.6)).toBe('-13%')
    expect(fmtSignedPct(0.2)).toBe('0%')
    expect(fmtSignedPct(null)).toBe('—')
  })
})

const res = (status: BenchResult['status']): BenchResult => ({ combo: {}, status })
const run = (id: string, modelId: string, startedAt: number, statuses: BenchResult['status'][], status: BenchmarkRun['status'] = 'running'): BenchmarkRun => ({
  id,
  modelId,
  modelName: modelId,
  label: '',
  startedAt,
  status,
  base,
  axes: [],
  prompt: { promptTokens: 2048, genTokens: 128, repeats: 2 },
  skipLikelyOom: true,
  results: statuses.map(res)
})

describe('progress and ETA', () => {
  it('counts combinations that are over', () => {
    expect(runProgress(run('a', 'm', 0, ['ok', 'failed', 'skipped', 'stopped', 'running', 'pending']))).toEqual({ done: 4, total: 6 })
  })
  it('estimates from the average time per finished combination', () => {
    const r = run('a', 'm', 1000, ['ok', 'ok', 'running', 'pending'])
    expect(etaMs(r, 1000 + 60_000)).toBe(60_000) // 30 s each, two left
  })
  it('has no estimate before the first one finishes, at the end, or when the run is over', () => {
    expect(etaMs(run('a', 'm', 0, ['running', 'pending']), 5000)).toBeNull()
    expect(etaMs(run('a', 'm', 0, ['ok', 'ok']), 5000)).toBeNull()
    expect(etaMs(run('a', 'm', 0, ['ok', 'pending'], 'stopped'), 5000)).toBeNull()
  })
})

describe('run list', () => {
  it('replaces a run by id or adds it, newest first', () => {
    const a = run('a', 'm', 100, ['ok'])
    const b = run('b', 'm', 200, ['ok'])
    expect(upsertRun([a], b).map((r) => r.id)).toEqual(['b', 'a'])
    const a2 = { ...a, label: 'x' }
    const out = upsertRun([b, a], a2)
    expect(out.map((r) => r.id)).toEqual(['b', 'a'])
    expect(out[1].label).toBe('x')
  })
  it('keeps one model’s runs', () => {
    const list = [run('a', 'm1', 1, []), run('b', 'm2', 2, []), run('c', 'm1', 3, [])]
    expect(runsForModel(list, 'm1').map((r) => r.id)).toEqual(['c', 'a'])
  })
})

describe('late replies', () => {
  const old = run('a', 'm', 100, ['pending', 'pending'])
  const mid = run('a', 'm', 100, ['ok', 'running'])
  const over = run('a', 'm', 100, ['ok', 'ok'], 'done')
  it('keeps a run that is further along over an older copy', () => {
    expect(mergeRun([over], old)).toEqual([over])
    expect(mergeRun([over], mid)[0]).toBe(over)
    expect(mergeRun([mid], old)[0]).toBe(mid)
  })
  it('takes a newer copy, and adds an unknown run', () => {
    expect(mergeRun([old], mid)[0]).toBe(mid)
    expect(mergeRun([mid], over)[0]).toBe(over)
    expect(mergeRun([], old)).toEqual([old])
  })
  it('does not let the start reply replace a run an event already brought', () => {
    expect(insertRunIfAbsent([over], old)[0]).toBe(over)
    expect(insertRunIfAbsent([], old)).toEqual([old])
  })
})
