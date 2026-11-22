import { describe, expect, it } from 'vitest'
import { DEFAULT_LOAD_CONFIG, DEFAULT_SAMPLING } from '@shared/defaults'
import { addPairs, BUILTIN_PROMPTS, cleanRun, newRun, nextOutput, nextPair, randomSeeds, scores, speedByCandidate } from '@shared/writingBench'

function fixture() {
  return newRun({ label: 'test', prompts: BUILTIN_PROMPTS.slice(0, 2), candidates: ['a', 'b'].map((id) => ({ id, label: id, modelId: id, modelName: id, adapters: [], sampling: DEFAULT_SAMPLING, load: null })), seeds: [1, 2], maxTokens: 600 }, '2026-01-01', 'run')
}
describe('writing benchmark', () => {
  it('orders outputs by candidate, prompt and seed and finds the first pending', () => {
    const run = fixture()
    expect(run.outputs.map((o) => o.key)).toEqual(['builtin-1|a|1', 'builtin-1|a|2', 'builtin-2|a|1', 'builtin-2|a|2', 'builtin-1|b|1', 'builtin-1|b|2', 'builtin-2|b|1', 'builtin-2|b|2'])
    run.outputs[0].status = 'done'
    expect(nextOutput(run)).toBe(run.outputs[1])
  })
  it('adds stable distinct pairs and finds the next unvoted pair', () => {
    const run = fixture(); run.outputs.forEach((o) => { o.status = 'done' })
    const paired = addPairs(run, () => 0)
    expect(paired.pairs).toHaveLength(4)
    expect(addPairs(paired, () => 1).pairs).toEqual(paired.pairs)
    expect(nextPair(paired)).toEqual(paired.pairs[0])
    expect(addPairs(run, () => 1).pairs.map((p) => p.id)).toEqual(paired.pairs.map((p) => p.id))
  })
  it('computes Elo in vote order including ties and ignores skips', () => {
    const run = fixture(); run.outputs.forEach((o) => { o.status = 'done' })
    const paired = addPairs(run, () => 0)
    paired.pairs[0].vote = 'a'; paired.pairs[0].votedAt = '1'
    paired.pairs[1].vote = 'tie'; paired.pairs[1].votedAt = '2'
    paired.pairs[2].vote = 'skip'; paired.pairs[2].votedAt = '3'
    const [a, b] = scores(paired)
    expect(a.rating).toBeCloseTo(1014.530498, 5); expect(b.rating).toBeCloseTo(985.469502, 5)
    expect(a).toMatchObject({ wins: 1, ties: 1, votes: 2 }); expect(b.losses).toBe(1)
  })
  it('makes distinct seeds even with repeated random values', () => { expect(randomSeeds(3, () => 0)).toEqual([0, 1, 2]) })
  it('rejects bad shapes and preserves unknown safe values', () => {
    const run = { ...fixture(), future: { kept: true } }
    expect(cleanRun(run)).toEqual({ ok: true, run })
    expect(cleanRun({ ...run, outputs: [] }).ok).toBe(false)
    expect(cleanRun({ ...run, id: '../bad' }).ok).toBe(false)
    expect(cleanRun({ ...run, candidates: [{ ...run.candidates[0], sampling: {} }] }).ok).toBe(false)
  })
  it('reads a candidate without load settings (an old run file) as the model saved settings', () => {
    const run = fixture()
    const old = { ...run, candidates: run.candidates.map(({ load: _load, ...rest }) => rest) }
    const cleaned = cleanRun(old)
    expect(cleaned.ok).toBe(true)
    if (cleaned.ok) expect(cleaned.run.candidates.map((c) => c.load)).toEqual([null, null])
  })
  it('keeps a candidate load config and rejects a malformed one', () => {
    const run = fixture()
    const load = { ...DEFAULT_LOAD_CONFIG, contextLength: 4096, gpuLayers: 20 }
    const withLoad = { ...run, candidates: [{ ...run.candidates[0], load }, run.candidates[1]] }
    expect(cleanRun(withLoad)).toEqual({ ok: true, run: withLoad })
    expect(cleanRun({ ...run, candidates: [{ ...run.candidates[0], load: { ...load, contextLength: 'big' } }, run.candidates[1]] }).ok).toBe(false)
    expect(cleanRun({ ...run, candidates: [{ ...run.candidates[0], load: 'x' }, run.candidates[1]] }).ok).toBe(false)
    // "Auto (fit)" context and empty draft tokens are stored as null and must still load next time.
    const auto = { ...run, candidates: [{ ...run.candidates[0], load: { ...load, contextLength: null, draftTokens: null } }, run.candidates[1]] }
    expect(cleanRun(auto).ok).toBe(true)
  })
  it('reads an old output without speed fields as null and rejects a bad speed', () => {
    const run = fixture()
    const old = { ...run, outputs: run.outputs.map(({ tokensPerSecond: _a, promptTokensPerSecond: _b, tokens: _c, ...rest }) => rest) }
    const cleaned = cleanRun(old)
    expect(cleaned.ok).toBe(true)
    if (cleaned.ok) expect(cleaned.run.outputs.every((o) => o.tokensPerSecond === null && o.promptTokensPerSecond === null && o.tokens === null)).toBe(true)
    const withSpeed = { ...run, outputs: run.outputs.map((o, i) => (i === 0 ? { ...o, tokensPerSecond: 12.5, promptTokensPerSecond: 300, tokens: 80 } : o)) }
    expect(cleanRun(withSpeed)).toEqual({ ok: true, run: withSpeed })
    for (const bad of [-1, Number.NaN, Infinity, '5'])
      expect(cleanRun({ ...run, outputs: run.outputs.map((o, i) => (i === 0 ? { ...o, tokensPerSecond: bad } : o)) }).ok).toBe(false)
    expect(cleanRun({ ...run, outputs: run.outputs.map((o, i) => (i === 0 ? { ...o, promptTokensPerSecond: -3 } : o)) }).ok).toBe(false)
    expect(cleanRun({ ...run, outputs: run.outputs.map((o, i) => (i === 0 ? { ...o, tokens: -3 } : o)) }).ok).toBe(false)
  })
  it('starts new outputs with null speeds', () => {
    expect(fixture().outputs.every((o) => o.tokensPerSecond === null && o.promptTokensPerSecond === null && o.tokens === null)).toBe(true)
  })
  it('averages speeds per candidate over done outputs that have the value', () => {
    const run = fixture()
    const set = (key: string, patch: Partial<(typeof run.outputs)[number]>): void => { Object.assign(run.outputs.find((o) => o.key === key)!, patch) }
    set('builtin-1|a|1', { status: 'done', tokensPerSecond: 10, promptTokensPerSecond: 100 })
    set('builtin-1|a|2', { status: 'done', tokensPerSecond: 20, promptTokensPerSecond: null })
    set('builtin-2|a|1', { status: 'failed', tokensPerSecond: 999 })
    set('builtin-1|b|1', { status: 'done', tokensPerSecond: null })
    expect(speedByCandidate(run)).toEqual([
      { candidateId: 'a', tokensPerSecond: 15, promptTokensPerSecond: 100, measured: 2 },
      { candidateId: 'b', tokensPerSecond: null, promptTokensPerSecond: null, measured: 0 }
    ])
  })
  it('weights speeds by tokens, so a cached prompt of a few tokens does not count like a full one', () => {
    const run = fixture()
    const a = run.outputs.filter((o) => o.candidateId === 'a')
    // First output: 1000 prompt tokens at 1000 t/s (1 s). Later: 5 tokens at 50 t/s (0.1 s), the cached prompt.
    Object.assign(a[0], { status: 'done', promptTokens: 1000, promptTokensPerSecond: 1000, tokens: 100, tokensPerSecond: 20 })
    Object.assign(a[1], { status: 'done', promptTokens: 5, promptTokensPerSecond: 50, tokens: 300, tokensPerSecond: 30 })
    const s = speedByCandidate(run)[0]
    expect(s.promptTokensPerSecond).toBeCloseTo(1005 / 1.1, 5)
    expect(s.tokensPerSecond).toBeCloseTo(400 / 15, 5)
    // Old outputs without counts: plain mean.
    Object.assign(a[0], { promptTokens: null, tokens: null }); Object.assign(a[1], { promptTokens: null, tokens: null })
    expect(speedByCandidate(run)[0]).toMatchObject({ promptTokensPerSecond: 525, tokensPerSecond: 25 })
  })
})
