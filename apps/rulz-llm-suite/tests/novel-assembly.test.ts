import { describe, expect, it } from 'vitest'
import { anchorProse, assembleRequest } from '../src/shared/novelAssembly'
import type { AssemblyEntry, AssemblyInput, AssemblySummary, Counted } from '../src/shared/novel'

// Scaffolding the assembler adds (see novelAssembly.ts): each item costs one token for the blank line
// in front of it, and each section its heading: "## Heading" + 2 characters, ceil(chars / 3).
const ITEM = 1
const HEAD = { bible: 6, story: 6, scene: 5, relevant: 6, prose: 7, after: 9, task: 5 }

const c = (text: string, tokens: number): Counted => ({ text, tokens })

function entry(id: string, tokens: number, over: Partial<AssemblyEntry> = {}): AssemblyEntry {
  const short = over.short ?? c(`${id} short`, Math.ceil(tokens / 4))
  return { ...c(`${id} full`, tokens), id, name: id, priority: 50, reason: 'always', fixed: true, ...over, short }
}

function summary(label: string, tokens: number, kind: 'chapter' | 'scene' = 'chapter'): AssemblySummary {
  return { ...c(`recap ${label}`, tokens), label, kind, fallback: 'none' }
}

// 4096 tokens of context: available = 4096 - 20 - 500 - 256 = 3320.
function input(over: Partial<AssemblyInput> = {}): AssemblyInput {
  return {
    action: 'continue',
    contextLength: 4096,
    templateOverhead: 20,
    budget: { replyTokens: 500, proseShare: 0.35, bibleShare: 0.3 },
    system: c('SYSTEM', 100),
    task: c('TASK', 50),
    scene: c('SCENE', 50),
    entries: [],
    summaries: [],
    proseBlocks: [],
    proseAfter: c('', 0),
    anchorStep: 100,
    ...over
  }
}

const blocks = (count: number, tokens = 100, from = 0): Counted[] =>
  Array.from({ length: count }, (_, i) => c(`paragraph ${from + i}`, tokens))

const userText = (input: AssemblyInput): string => assembleRequest(input).messages[1].content

describe('anchorProse', () => {
  it('returns 0 for no blocks or a tail that fits', () => {
    expect(anchorProse([], 100, 50)).toBe(0)
    expect(anchorProse(blocks(5, 10), 50, 20)).toBe(0)
    expect(anchorProse(blocks(5, 10), 500, 20)).toBe(0)
  })

  it('moves in whole steps when blocks are appended', () => {
    // 10-token blocks, budget 500, step 100: anchor points every 10 blocks.
    const all = blocks(100, 10)
    const starts: number[] = []
    for (let n = 1; n <= 100; n++) starts.push(anchorProse(all.slice(0, n), 500, 100))
    for (let n = 1; n <= 50; n++) expect(starts[n - 1]).toBe(0)
    for (let n = 51; n <= 60; n++) expect(starts[n - 1]).toBe(10)
    for (let n = 61; n <= 70; n++) expect(starts[n - 1]).toBe(20)
    for (let n = 71; n <= 80; n++) expect(starts[n - 1]).toBe(30)
    for (let n = 81; n <= 90; n++) expect(starts[n - 1]).toBe(40)
    for (let n = 91; n <= 100; n++) expect(starts[n - 1]).toBe(50)
  })

  it('never moves backwards, only moves when the tail stopped fitting, and the tail fits', () => {
    // Deterministic pseudo-random block sizes (20-79) and several budgets and steps. Steps are at
    // least as big as a block and the budget at least as big as a step plus a block, as in real use.
    let seed = 12345
    const next = (): number => {
      seed = (seed * 1103515245 + 12345) % 2147483648
      return seed
    }
    for (const [budget, step] of [[300, 80], [900, 250], [1500, 400], [400, 100], [600, 150]]) {
      const all = Array.from({ length: 150 }, () => c('x', 20 + (next() % 60)))
      let previous = 0
      for (let n = 1; n <= all.length; n++) {
        const part = all.slice(0, n)
        const start = anchorProse(part, budget, step)
        expect(start).toBeGreaterThanOrEqual(previous)
        const tail = part.slice(start).reduce((total, block) => total + block.tokens, 0)
        if (start > previous) {
          // It moved: the old start no longer fitted.
          const oldTail = part.slice(previous).reduce((total, block) => total + block.tokens, 0)
          expect(oldTail).toBeGreaterThan(budget)
        }
        // A tail over budget is only allowed for the last block alone.
        if (tail > budget) expect(start).toBe(n - 1)
        previous = start
      }
    }
  })

  it('keeps the start while only the tail grows within budget', () => {
    const base = blocks(8, 100)
    const before = anchorProse(base, 1000, 200)
    expect(anchorProse([...base, c('more', 100)], 1000, 200)).toBe(before)
    expect(anchorProse([...base, c('more', 100), c('and more', 100)], 1000, 200)).toBe(before)
  })

  it('falls back to the earliest block that fits when no anchor point does', () => {
    // Step bigger than all the text: the only anchor point is block 0.
    expect(anchorProse(blocks(10, 100), 250, 1000)).toBe(8)
  })

  it('keeps the last block even when it alone is over budget', () => {
    expect(anchorProse([c('a', 10), c('b', 10), c('c', 500)], 100, 50)).toBe(2)
    expect(anchorProse(blocks(4, 10), 0, 50)).toBe(3)
  })

  it('treats a step below 1 as 1 and a negative budget as 0', () => {
    expect(anchorProse(blocks(4, 10), 25, 0)).toBe(2)
    expect(anchorProse(blocks(4, 10), -5, -5)).toBe(3)
  })
})

describe('assembleRequest budget', () => {
  it('computes available = context - overhead - reply - max(256, 3%)', () => {
    expect(assembleRequest(input()).available).toBe(3320)
    // 3% of 100000 is 3000, more than the 256 floor.
    expect(
      assembleRequest(input({ contextLength: 100000, templateOverhead: 100, budget: { replyTokens: 1000, proseShare: 0.35, bibleShare: 0.3 } }))
        .available
    ).toBe(100000 - 100 - 1000 - 3000)
    expect(assembleRequest(input({ contextLength: 8192, templateOverhead: 0 })).available).toBe(8192 - 500 - 256)
  })

  it('counts the headings and separators it adds in every part and in totalTokens', () => {
    const result = assembleRequest(input({ proseBlocks: blocks(20) }))
    const part = (kind: string) => result.parts.find((p) => p.kind === kind)
    expect(part('system')?.tokens).toBe(100)
    expect(part('task')?.tokens).toBe(50 + HEAD.task + ITEM)
    expect(part('scene')?.tokens).toBe(50 + HEAD.scene + ITEM)
    expect(part('prose')?.tokens).toBe(20 * (100 + ITEM) + HEAD.prose)
    const content = result.parts.reduce((total, p) => total + p.tokens, 0)
    expect(content).toBe(100 + 56 + 56 + 2027)
    expect(result.totalTokens).toBe(content + 20)
  })

  it('estimates the scaffolding at no less than the real characters / 3', () => {
    const result = assembleRequest(
      input({
        entries: [entry('a', 100), entry('t', 100, { fixed: false })],
        summaries: [summary('Chapter 1', 100)],
        proseBlocks: blocks(5),
        proseAfter: c('after', 10)
      })
    )
    const itemTokens = 100 + 100 + 100 + 5 * 100 + 10 + 100 + 50 + 50
    const body = result.messages[1].content
    const itemChars = ['a full', 't full', 'recap Chapter 1', 'SCENE', 'TASK', 'after']
      .concat(blocks(5).map((b) => b.text))
      .reduce((total, text) => total + text.length, 0)
    const scaffoldChars = body.length - itemChars
    const scaffoldTokens = result.parts.reduce((total, p) => total + p.tokens, 0) - itemTokens
    expect(scaffoldTokens).toBeGreaterThanOrEqual(Math.ceil(scaffoldChars / 3))
  })

  it('warns when the system text or task and scene are large', () => {
    expect(assembleRequest(input()).warnings).toEqual([])
    // 30% of 3320 is 996; 15% is 498.
    expect(assembleRequest(input({ system: c('S', 1000) })).warnings.join(' ')).toContain('30%')
    expect(assembleRequest(input({ system: c('S', 996) })).warnings.join(' ')).not.toContain('30%')
    expect(assembleRequest(input({ task: c('T', 300), scene: c('S', 300) })).warnings.join(' ')).toContain('15%')
  })

  it('returns a result with a warning when even the minimum does not fit', () => {
    const result = assembleRequest(input({ contextLength: 700, proseBlocks: [c('text', 800)] }))
    expect(result.available).toBe(700 - 20 - 500 - 256)
    expect(result.warnings.join(' ')).toContain('do not fit')
    expect(result.totalTokens).toBeGreaterThan(result.available)
    expect(result.messages).toHaveLength(2)
    expect(result.messages[1].content).toContain('text')
  })

  it('keeps the newest block and warns when it is too big for the room left', () => {
    const result = assembleRequest(input({ contextLength: 2000, budget: { replyTokens: 200, proseShare: 0.35, bibleShare: 0.3 }, templateOverhead: 0, proseBlocks: [c('huge', 1400)] }))
    expect(result.available).toBe(1544)
    expect(result.proseStart).toBe(0)
    expect(result.warnings.join(' ')).toContain('larger than the available')
    expect(result.totalTokens).toBeGreaterThan(result.available)
  })

  it('never exceeds the available context when the pieces can be split', () => {
    for (let n = 1; n <= 120; n++) {
      const result = assembleRequest(input({ proseBlocks: blocks(n, 90) }))
      expect(result.totalTokens - 20).toBeLessThanOrEqual(result.available)
      expect(result.warnings).toEqual([])
    }
  })
})

describe('assembleRequest fill order', () => {
  it('reserves at least 800 tokens for prose before story so far', () => {
    // available = 2544, base 212, left 2332. proseShare 0.1 would give 233, the floor makes it 800, so
    // a 1600-token summary (1607 with scaffolding) does not fit in the 1532 left.
    const common = { contextLength: 3000, templateOverhead: 0, budget: { replyTokens: 200, proseShare: 0.1, bibleShare: 0.3 }, proseBlocks: blocks(30) }
    const result = assembleRequest(input({ ...common, summaries: [summary('Chapter 1', 1600)] }))
    expect(result.available).toBe(2544)
    expect(result.dropped).toEqual([
      { kind: 'storySoFar', label: 'Chapter 1', how: 'dropped', reason: 'budget' },
      expect.objectContaining({ kind: 'prose' })
    ])
    expect(result.parts.some((p) => p.kind === 'storySoFar')).toBe(false)
  })

  it('reserves only the prose there is when it is under 800 tokens', () => {
    // 3 blocks = 310 tokens with scaffolding; 2332 - 310 = 2022 left, so a 1900-token summary fits.
    const result = assembleRequest(
      input({
        contextLength: 3000,
        templateOverhead: 0,
        budget: { replyTokens: 200, proseShare: 0.1, bibleShare: 0.3 },
        proseBlocks: blocks(3),
        summaries: [summary('Chapter 1', 1900)]
      })
    )
    expect(result.dropped).toEqual([])
    expect(result.parts.find((p) => p.kind === 'storySoFar')?.tokens).toBe(HEAD.story + 1900 + ITEM)
    expect(result.proseStart).toBe(0)
  })

  it('gives unused room back to the prose tail', () => {
    // available 7416, base 212: the proseShare cap is 2521 but nothing else needs room, so all 40
    // blocks (4047 with scaffolding) go in.
    const result = assembleRequest(input({ contextLength: 8192, proseBlocks: blocks(40) }))
    expect(result.available).toBe(7416)
    expect(result.proseStart).toBe(0)
    const prose = result.parts.find((p) => p.kind === 'prose')
    expect(prose?.tokens).toBe(4047)
    expect(prose?.cap).toBe(7416 - 212)
    // With much more prose than room the tail starts later and still fits.
    const long = assembleRequest(input({ contextLength: 8192, proseBlocks: blocks(100) }))
    expect(long.proseStart).toBeGreaterThan(0)
    expect(long.parts.find((p) => p.kind === 'prose')!.tokens).toBeLessThanOrEqual(7416 - 212)
    expect(long.dropped.some((d) => d.kind === 'prose' && d.how === 'cut')).toBe(true)
  })

  it('counts prose after the cursor against the tail budget', () => {
    const result = assembleRequest(input({ contextLength: 8192, proseBlocks: blocks(100), proseAfter: c('AFTER TEXT', 500) }))
    expect(result.messages[1].content).toContain('## Text after the cursor\n\nAFTER TEXT')
    const prose = result.parts.find((p) => p.kind === 'prose')!
    expect(prose.tokens).toBeLessThanOrEqual(7416 - 212)
    expect(prose.tokens).toBeGreaterThan(500 + HEAD.after)
  })

  it('cuts the lowest priority entries to their first paragraph first', () => {
    // bible cap = floor(3320 * 0.3) = 996. Three 400-token entries cost 6 + 3 * 401 = 1209: cutting
    // the lowest priority one (to 100) gives 909.
    const entries = [entry('a', 400, { priority: 90 }), entry('b', 400, { priority: 50 }), entry('c', 400, { priority: 10 })]
    const result = assembleRequest(input({ entries }))
    expect(result.dropped).toEqual([{ kind: 'fixedEntries', label: 'c', how: 'cut', reason: 'budget' }])
    expect(result.includedEntries).toEqual(['a', 'b', 'c'])
    const body = result.messages[1].content
    expect(body).toContain('a full')
    expect(body).toContain('b full')
    expect(body).toContain('c short')
    expect(body).not.toContain('c full')
    expect(result.parts.find((p) => p.kind === 'fixedEntries')).toEqual({ kind: 'fixedEntries', tokens: HEAD.bible + 401 + 401 + 101, cap: 996 })
  })

  it('drops the lowest priority entries when cutting is not enough', () => {
    // Short versions as long as the full text cannot be cut, so the lowest priority one goes.
    const same = (id: string, priority: number) => entry(id, 400, { priority, short: c(`${id} full`, 400) })
    const result = assembleRequest(input({ entries: [same('a', 90), same('b', 50), same('c', 10)] }))
    expect(result.dropped).toEqual([{ kind: 'fixedEntries', label: 'c', how: 'dropped', reason: 'budget' }])
    expect(result.includedEntries).toEqual(['a', 'b'])
    expect(result.messages[1].content).not.toContain('c full')
  })

  it('cuts every entry it can before dropping any, then drops from the lowest priority', () => {
    // Short versions of 380 tokens: all cut = 6 + 3 * 381 = 1149 > 996, so c is dropped (768).
    const e = (id: string, priority: number) => entry(id, 400, { priority, short: c(`${id} short`, 380) })
    const result = assembleRequest(input({ entries: [e('a', 90), e('b', 50), e('c', 10)] }))
    // The drop frees room, so a and b get their full text back (see the next test).
    expect(result.dropped).toEqual([{ kind: 'fixedEntries', label: 'c', how: 'dropped', reason: 'budget' }])
    expect(result.includedEntries).toEqual(['a', 'b'])
  })

  it('lets fixed entries use the shared bible share first and triggered entries get the rest', () => {
    // cap 996. Fixed: 6 + 701 = 707, so triggered get 289: a 400-token entry is cut to 100 (107).
    const entries = [entry('fixed', 700), entry('trig', 400, { fixed: false, reason: 'trigger: x' })]
    const result = assembleRequest(input({ entries }))
    expect(result.dropped).toEqual([{ kind: 'triggeredEntries', label: 'trig', how: 'cut', reason: 'budget' }])
    expect(result.messages[1].content).toContain('fixed full')
    expect(result.messages[1].content).toContain('trig short')
    const triggered = result.parts.find((p) => p.kind === 'triggeredEntries')!
    expect(triggered.cap).toBe(996 - (HEAD.bible + 701))
  })

  it('drops triggered entries lowest priority first', () => {
    const same = (id: string, priority: number) =>
      entry(id, 150, { priority, fixed: false, reason: 'trigger: x', short: c(`${id} full`, 150) })
    const entries = [entry('fixed', 700), same('hi', 80), same('lo', 10)]
    const result = assembleRequest(input({ entries }))
    expect(result.dropped).toEqual([{ kind: 'triggeredEntries', label: 'lo', how: 'dropped', reason: 'budget' }])
    expect(result.includedEntries).toEqual(['fixed', 'hi'])
  })

  it('drops the oldest summaries first', () => {
    // available 1544, base 212: 1332 left. 6 + 3 * 501 = 1509 does not fit; without the oldest 1008 does.
    const summaries = [summary('old', 500), summary('mid', 500), summary('new', 500)]
    const result = assembleRequest(input({ contextLength: 2000, templateOverhead: 0, budget: { replyTokens: 200, proseShare: 0.35, bibleShare: 0.3 }, summaries }))
    expect(result.dropped).toEqual([{ kind: 'storySoFar', label: 'old', how: 'dropped', reason: 'budget' }])
    const body = result.messages[1].content
    expect(body).not.toContain('recap old')
    expect(body.indexOf('recap mid')).toBeLessThan(body.indexOf('recap new'))
    expect(result.parts.find((p) => p.kind === 'storySoFar')?.tokens).toBe(HEAD.story + 2 * 501)
  })

  it('drops every summary and entry rather than the minimum prose', () => {
    const result = assembleRequest(
      input({
        contextLength: 1500,
        templateOverhead: 0,
        budget: { replyTokens: 200, proseShare: 0.35, bibleShare: 0.3 },
        entries: [entry('a', 600)],
        summaries: [summary('s', 600)],
        proseBlocks: blocks(10)
      })
    )
    expect(result.available).toBe(1044)
    expect(result.includedEntries).toEqual([])
    expect(result.dropped.map((d) => d.kind).sort()).toEqual(['fixedEntries', 'prose', 'storySoFar'])
    expect(result.warnings).toEqual([])
  })
})

describe('assembleRequest messages', () => {
  const full = (): AssemblyInput =>
    input({
      entries: [
        entry('zed', 100, { priority: 50, name: 'Zed' }),
        entry('amy', 100, { priority: 50, name: 'Amy' }),
        entry('top', 100, { priority: 90, name: 'Top' }),
        entry('mara', 100, { priority: 70, name: 'Mara', fixed: false, reason: 'trigger: witch' })
      ],
      summaries: [summary('Chapter 1: The ferry', 50), summary('Scene: Night crossing', 50, 'scene')],
      proseBlocks: blocks(3),
      proseAfter: c('AFTER', 10)
    })

  it('puts a system message first and one user message with the sections in a stable order', () => {
    const result = assembleRequest(full())
    expect(result.messages.map((m) => m.role)).toEqual(['system', 'user'])
    expect(result.messages[0].content).toBe('SYSTEM')
    const body = result.messages[1].content
    const headings = ['## Story bible', '## Story so far', '## This scene', '## Also relevant', '## The text so far', '## Text after the cursor', '## Your task']
    const at = headings.map((h) => body.indexOf(h))
    expect(at.every((i) => i >= 0)).toBe(true)
    expect([...at].sort((a, b) => a - b)).toEqual(at)
    expect(body.endsWith('## Your task\n\nTASK')).toBe(true)
  })

  it('orders entries fixed first by priority, then name, and summaries oldest first', () => {
    const result = assembleRequest(full())
    const body = result.messages[1].content
    expect(result.includedEntries).toEqual(['top', 'amy', 'zed', 'mara'])
    expect(body.indexOf('top full')).toBeLessThan(body.indexOf('amy full'))
    expect(body.indexOf('amy full')).toBeLessThan(body.indexOf('zed full'))
    expect(body.indexOf('recap Chapter 1')).toBeLessThan(body.indexOf('recap Scene'))
    // Triggered entries sit after the scene section, not in the bible.
    expect(body.indexOf('mara full')).toBeGreaterThan(body.indexOf('## This scene'))
  })

  it('omits empty sections', () => {
    const body = userText(input())
    expect(body).toBe('## This scene\n\nSCENE\n\n## Your task\n\nTASK')
    expect(userText(input({ proseBlocks: blocks(1) }))).toContain('## The text so far\n\nparagraph 0')
    expect(userText(input({ proseBlocks: blocks(1) }))).not.toContain('Text after the cursor')
    expect(userText(input({ scene: c('', 0), task: c('', 0) }))).toBe('')
  })

  it('keeps the system message and the sections before the prose identical when only the prose grows', () => {
    const before = (n: number): string[] => {
      const data = full()
      data.proseBlocks = blocks(n)
      data.proseAfter = c('', 0)
      const result = assembleRequest(data)
      const body = result.messages[1].content
      return [result.messages[0].content, body.slice(0, body.indexOf('## The text so far'))]
    }
    const first = before(5)
    for (const n of [6, 9, 20, 60]) expect(before(n)).toEqual(first)
  })

  it('keeps the same prefix and a stable start across many appended blocks, moving in steps', () => {
    const prefixes = new Set<string>()
    const starts: number[] = []
    for (let n = 1; n <= 150; n++) {
      const data = full()
      data.proseBlocks = blocks(n, 90)
      data.proseAfter = c('', 0)
      data.anchorStep = 800
      const result = assembleRequest(data)
      const body = result.messages[1].content
      prefixes.add(body.slice(0, body.indexOf('## The text so far')))
      starts.push(result.proseStart)
    }
    expect(prefixes.size).toBe(1)
    for (let i = 1; i < starts.length; i++) expect(starts[i]).toBeGreaterThanOrEqual(starts[i - 1])
    const moves = starts.filter((start, i) => i > 0 && start !== starts[i - 1]).length
    // 150 blocks (13650 tokens) past a ~3000-token tail in 800-token steps: about 13 moves.
    expect(moves).toBeGreaterThan(5)
    expect(moves).toBeLessThan(20)
  })

  it('returns parts only for the kinds present, in fill order, with caps', () => {
    const result = assembleRequest(full())
    expect(result.parts.map((p) => p.kind)).toEqual(['system', 'task', 'scene', 'prose', 'fixedEntries', 'triggeredEntries', 'storySoFar'])
    expect(result.parts.find((p) => p.kind === 'system')?.cap).toBeNull()
    expect(result.parts.find((p) => p.kind === 'fixedEntries')?.cap).toBe(996)
    expect(assembleRequest(input()).parts.map((p) => p.kind)).toEqual(['system', 'task', 'scene'])
  })
})

describe('assembleRequest review fixes', () => {
  it('restores full entry text, highest priority first, after drops free room', () => {
    // cap 996: all cut = 1149 > 996, c is dropped (768); then a and b fit with full text (808).
    const e = (id: string, priority: number) => entry(id, 400, { priority, short: c(`${id} short`, 380) })
    const result = assembleRequest(input({ entries: [e('a', 90), e('b', 50), e('c', 10)] }))
    const body = result.messages[1].content
    expect(body).toContain('a full')
    expect(body).toContain('b full')
    expect(body).not.toContain('short')
    expect(result.parts.find((p) => p.kind === 'fixedEntries')?.tokens).toBe(HEAD.bible + 2 * 401)
    // Only part of the room comes back: a full (401) fits next to b short, b stays cut.
    const tight = assembleRequest(
      input({
        budget: { replyTokens: 500, proseShare: 0.35, bibleShare: 0.25 },
        entries: [e('a', 90), e('b', 50), e('c', 10)]
      })
    )
    // cap 830: all cut 1149 -> drop c 768 -> a full 788 fits -> b full 808 fits.
    expect(tight.includedEntries).toEqual(['a', 'b'])
    const smaller = assembleRequest(
      input({
        budget: { replyTokens: 500, proseShare: 0.35, bibleShare: 0.24 },
        entries: [e('a', 90), e('b', 50), e('c', 10)]
      })
    )
    // cap 796: drop c 768 -> a full 788 fits -> b full 808 does not, b stays cut.
    expect(smaller.messages[1].content).toContain('a full')
    expect(smaller.messages[1].content).toContain('b short')
    expect(smaller.dropped).toContainEqual({ kind: 'fixedEntries', label: 'b', how: 'cut', reason: 'budget' })
  })

  it('keeps the story so far the same while the prose grows from 200 tokens past its cap', () => {
    const summaries = [summary('Chapter 1', 300), summary('Chapter 2', 300)]
    const stories = new Set<string>()
    const kept = new Set<number>()
    for (let n = 2; n <= 90; n++) {
      const result = assembleRequest(input({ contextLength: 3000, templateOverhead: 0, summaries, proseBlocks: blocks(n), anchorStep: 400 }))
      const body = result.messages[1].content
      stories.add(body.slice(0, body.indexOf('## This scene')))
      kept.add(result.dropped.filter((d) => d.kind === 'storySoFar').length)
    }
    expect(stories.size).toBe(1)
    expect(kept).toEqual(new Set([0]))
  })

  it('keeps a big block that straddles a grid boundary as a start', () => {
    expect(anchorProse([c('a', 200), c('b', 600), c('c', 50)], 800, 250)).toBe(1)
  })

  it('caps the step at half the budget', () => {
    // Step 5000 would leave only block 0; capped to 100 there are anchor points every block.
    expect(anchorProse(blocks(10, 100), 450, 5000)).toBe(6)
  })
})
