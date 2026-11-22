/// <reference lib="dom" />
// Stage 3 renderer logic: state groups and voice targets in requests, Check findings, the Update state card,
// State tab ops, the character interview and voice notes. IPC is mocked like in renderer-novel.test.ts.
declare global {
  interface Window {
    rulz: import('@shared/ipc').RulzApi
  }
}

import { ChangeSet } from '@codemirror/state'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { BibleEntry, CheckFinding, NovelOutline, NovelProjectState, StateFact, StateOp } from '@shared/novel'
import { findingKey } from '@shared/novelChecks'
import { defaultProject } from '@shared/novelPrompts'
import {
  buildRequest,
  factAddOp,
  factEditOps,
  factValues,
  fitInterview,
  interviewKey,
  interviewSummaries,
  makeFindingsEntry,
  resolveStateChanges,
  stateOpsFor,
  visibleFindings,
  voiceNoteBlock,
  wholeSceneProblem,
  type RunContext
} from '../src/renderer/src/lib/novelRun'
import { locateMarks, mapMarks } from '../src/renderer/src/lib/novelEditor'

vi.mock('sonner', () => ({ toast: { error: vi.fn(), message: vi.fn(), success: vi.fn(), warning: vi.fn() } }))

const entry = (id: string, name: string, extra: Partial<BibleEntry> = {}): BibleEntry => ({
  id,
  name,
  kind: 'character',
  aliases: [],
  triggers: [],
  priority: 50,
  always: false,
  body: `${name} is a person.\n\nMore about ${name}.`,
  problems: [],
  ...extra
})

const scene = (id: string, title: string, extra: Record<string, unknown> = {}) => ({
  id,
  title,
  file: `${id}.md`,
  beats: `Beats of ${title}`,
  summary: '',
  summaryOf: '',
  status: 'draft' as const,
  pov: null as string | null,
  present: [] as string[],
  sticky: [] as string[],
  targetWords: null as number | null,
  ...extra
})

const bible = (): BibleEntry[] => [
  entry('mara', 'Mara', { body: 'Mara keeps the light.\n\nShe distrusts everyone.\n\n## Voice\n- clipped, never swears' }),
  entry('oskar', 'Oskar', { body: 'Oskar sails the ferry.\n\n## Voice\n- talks too much' })
]

const outline = (): NovelOutline => ({
  format: 1,
  chapters: [
    {
      id: 'c1',
      title: 'One',
      act: '',
      summary: '',
      summaryOf: '',
      scenes: [scene('s1', 'Ferry', { pov: 'mara', present: ['oskar'], summary: 'Mara crosses with Oskar.' }), scene('s2', 'Dock', { pov: 'mara', present: ['oskar'] }), scene('s3', 'Inn', { pov: 'oskar' })]
    }
  ]
})

const fact = (id: string, extra: Partial<StateFact> = {}): StateFact => ({ id, entity: 'mara', kind: 'location', value: 'the lighthouse', about: null, from: 's1', until: null, ...extra })

const count = async (t: string): Promise<number> => Math.ceil(t.length / 4)

const SCENE_TEXT = 'Mara stood at the lighthouse.\n\n"I never met Oskar," she said.\n\nThe lamp turned.'

function ctx(over: Partial<RunContext> = {}): RunContext {
  return {
    action: 'dialogue',
    project: defaultProject('Test'),
    style: 'Plain.',
    outline: outline(),
    bible: bible(),
    sceneId: 's2',
    sceneHashes: {},
    text: SCENE_TEXT,
    cursor: SCENE_TEXT.length,
    selection: null,
    note: '',
    contextLength: 8192,
    templateOverhead: 20,
    facts: [fact('f1'), fact('f2', { entity: 'oskar', kind: 'carries', value: 'a brass key' })],
    ...over
  }
}

describe('requests: state groups and voice targets', () => {
  it('builds one counted state group per entity and marks the Dialogue voice targets', async () => {
    const built = await buildRequest(ctx(), count)
    const groups = built.input.stateGroups ?? []
    expect(groups.map((g) => g.entity).sort()).toEqual(['mara', 'oskar'])
    const mara = groups.find((g) => g.entity === 'mara')!
    // Lines only (the assembler adds the heading), counted with the same counter as everything else.
    expect(mara.text).toContain('location: the lighthouse')
    expect(mara.text).not.toContain('###')
    expect(mara.tokens).toBe(await count(mara.text))
    const entries = built.input.entries
    const maraEntry = entries.find((e) => e.id === 'mara')!
    expect(maraEntry.voice).toBe(true)
    expect(entries.find((e) => e.id === 'oskar')?.voice).toBe(true)
    // The short form keeps the voice section.
    expect(maraEntry.short.text).toContain('clipped, never swears')
    expect(maraEntry.short.tokens).toBe(await count(maraEntry.short.text))
  })

  it('marks no voice target for Continue without a selection, and leaves state out when there are no facts', async () => {
    const built = await buildRequest(ctx({ action: 'continue', facts: [] }), count)
    expect(built.input.entries.some((e) => e.voice)).toBe(false)
    expect(built.input.stateGroups).toBeUndefined()
  })

  it('marks characters named in the selection as voice targets', async () => {
    const text = 'Alpha. Oskar laughed. Omega.'
    const from = text.indexOf('Oskar')
    const built = await buildRequest(ctx({ action: 'rewrite', text, cursor: from, selection: { from, to: from + 12 }, variant: 'tighter' }), count)
    expect(built.input.entries.filter((e) => e.voice).map((e) => e.id)).toContain('oskar')
  })

  it('Update state sends the state with ids, what is recorded here, and the whole scene', async () => {
    const facts = [fact('f1'), fact('f9', { value: 'the dock', from: 's2' }), fact('f3', { kind: 'condition', value: 'wounded', from: 's1', until: 's2' })]
    const built = await buildRequest(ctx({ action: 'updateState', facts }), count)
    const task = built.input.task.text
    expect(task).toContain('[f1] Mara — location: the lighthouse')
    expect(task).toContain('[f9] Mara — location: the dock')
    expect(task).toContain('(ended here)')
    expect(built.input.proseBlocks.map((b) => b.text).join('\n\n')).toBe(SCENE_TEXT)
    expect(built.input.scene.text).toContain('id: mara')
  })
})

describe('whole-scene refusal', () => {
  it('is refused when not all prose fits (proseStart > 0) or the assembler says the whole scene does not fit', async () => {
    const long = Array.from({ length: 400 }, (_, i) => `Paragraph ${i} has some words in it to fill the context.`).join('\n\n')
    const built = await buildRequest(ctx({ action: 'checkScene', text: long, cursor: long.length, contextLength: 2048 }), count)
    expect(built.result.proseStart > 0 || built.result.warnings.some((w) => /whole scene/i.test(w))).toBe(true)
    expect(wholeSceneProblem(built)).not.toBeNull()
    const ok = await buildRequest(ctx({ action: 'checkScene' }), count)
    expect(wholeSceneProblem(ok)).toBeNull()
  })
})

describe('check findings', () => {
  const f = (type: CheckFinding['type'], quote: string): CheckFinding => ({ type, quote, problem: 'p', conflictsWith: 'c', suggestion: 's' })

  it('keeps the text hash, maps quotes to ranges and flags the ones not found', () => {
    const entryA = makeFindingsEntry(SCENE_TEXT, [f('continuity', '"I never met Oskar,"'), f('tense', 'not in the scene'), f('continuity', '"I never met Oskar,"')])
    // The repeat is one finding.
    expect(entryA.items).toHaveLength(2)
    expect(entryA.located[findingKey(entryA.items[0])]).toBe(true)
    expect(entryA.located[findingKey(entryA.items[1])]).toBe(false)
    const marks = locateMarks(SCENE_TEXT, entryA.items.map((x) => ({ id: findingKey(x), type: x.type, quote: x.quote })))
    expect(marks).toHaveLength(1)
    expect(SCENE_TEXT.slice(marks[0].from, marks[0].to)).toBe('"I never met Oskar,"')
  })

  it('hides dismissed findings', () => {
    const items = [f('continuity', 'a'), f('pov', 'b')]
    expect(visibleFindings(items, [findingKey(items[0])])).toEqual([items[1]])
    expect(visibleFindings(items, undefined)).toEqual(items)
  })

  it('moves marks with edits and drops a mark whose text was deleted', () => {
    const marks = [{ id: 'k', type: 'pov' as const, from: 10, to: 20 }]
    const before = ChangeSet.of({ from: 0, insert: 'XXXXX' }, 40)
    expect(mapMarks(marks, before)).toEqual([{ id: 'k', type: 'pov', from: 15, to: 25 }])
    const deleted = ChangeSet.of({ from: 8, to: 22 }, 40)
    expect(mapMarks(marks, deleted)).toEqual([])
    // Typing at an edge stays outside the mark.
    const edge = ChangeSet.of({ from: 20, insert: 'tail' }, 40)
    expect(mapMarks(marks, edge)).toEqual([{ id: 'k', type: 'pov', from: 10, to: 20 }])
  })
})

describe('update-state card logic', () => {
  const before = [fact('f1'), fact('f2', { entity: 'oskar', kind: 'carries', value: 'a brass key' })]

  it('leaves unresolved names and unknown fact ids out and says why', () => {
    const r = resolveStateChanges(
      [
        { op: 'add', entity: 'Mara', kind: 'carries', value: 'the  lamp\noil', about: null },
        { op: 'add', entity: 'Nobody', kind: 'knows', value: 'a secret', about: null },
        { op: 'end', factId: 'f1', reason: 'she left' },
        { op: 'end', factId: 'f404', reason: 'x' },
        { op: 'add', entity: 'Oskar', kind: 'relationship', value: 'distrusts her', about: 'Mara' }
      ],
      bible(),
      before
    )
    expect(r.items.map((i) => i.planned)).toEqual([
      { op: 'add', entity: 'mara', kind: 'carries', value: 'the lamp oil', about: null },
      { op: 'end', factId: 'f1' },
      { op: 'add', entity: 'oskar', kind: 'relationship', value: 'distrusts her', about: 'mara' }
    ])
    expect(r.left).toHaveLength(2)
    expect(r.left.join('|')).toContain('Nobody')
    expect(r.left.join('|')).toContain('f404')
  })

  it('sends a relationship without a partner as about null, for main to reject', () => {
    const r = resolveStateChanges([{ op: 'add', entity: 'Mara', kind: 'relationship', value: 'trusts', about: null }], bible(), before)
    expect(r.items[0].planned).toMatchObject({ op: 'add', about: null })
  })

  it('makes ops for the accepted changes only, from and until at this scene', () => {
    const r = resolveStateChanges([{ op: 'add', entity: 'Mara', kind: 'carries', value: 'a key', about: null }, { op: 'end', factId: 'f1', reason: '' }], bible(), before)
    r.items[1].accepted = false
    let n = 0
    const { ops, labels } = stateOpsFor(r.items, 's2', () => `fnew${++n}`)
    expect(ops).toEqual([{ op: 'add', fact: { id: 'fnew1', entity: 'mara', kind: 'carries', value: 'a key', about: null, from: 's2', until: null } }])
    expect(labels).toHaveLength(1)
    r.items[1].accepted = true
    expect(stateOpsFor(r.items, 's2').ops[1]).toEqual({ op: 'end', factId: 'f1', until: 's2' })
  })

  it('builds the State tab edit and add ops', () => {
    const f = fact('f1')
    const v = factValues(f)
    expect(factEditOps(f, v)).toEqual([])
    expect(factEditOps(f, { ...v, value: ' the harbour ', until: 's3' })).toEqual([{ op: 'edit', factId: 'f1', changes: { value: 'the harbour', until: 's3' } }])
    // Leaving the relationship kind clears about.
    const rel = fact('f5', { kind: 'relationship', about: 'oskar', value: 'distrusts' })
    expect(factEditOps(rel, { ...factValues(rel), kind: 'knows' })).toEqual([{ op: 'edit', factId: 'f5', changes: { kind: 'knows', about: null } }])
    expect(factAddOp({ entity: 'mara', kind: 'knows', value: ' a secret ', about: 'oskar', from: 's1', until: '' }, () => 'fx')).toEqual({
      op: 'add',
      fact: { id: 'fx', entity: 'mara', kind: 'knows', value: 'a secret', about: null, from: 's1', until: null }
    })
  })
})

describe('interview logic', () => {
  it('keys history by character and scene', () => {
    expect(interviewKey('mara', 's1')).not.toBe(interviewKey('mara', 's2'))
    expect(interviewKey('mara', 's1')).not.toBe(interviewKey('oskar', 's1'))
  })

  it('summarises only earlier scenes where the character is POV or present', () => {
    const o = outline()
    o.chapters[0].scenes[1].summary = 'Dock summary.'
    o.chapters[0].scenes[2].summary = 'Inn summary.'
    expect(interviewSummaries(o, 'mara', 's3')).toEqual(['Ferry: Mara crosses with Oskar.', 'Dock: Dock summary.'])
    expect(interviewSummaries(o, 'oskar', 's2')).toEqual(['Ferry: Mara crosses with Oskar.'])
    expect(interviewSummaries(o, 'mara', 's1')).toEqual([])
  })

  it('drops the oldest turns to fit and never the system prompt', async () => {
    const turns = Array.from({ length: 6 }, (_, i) => ({ role: i % 2 === 0 ? ('user' as const) : ('assistant' as const), content: `turn ${i} ${'x'.repeat(396)}` }))
    const countWhole = async (m: { content: string }[]): Promise<number> => m.reduce((n, x) => n + Math.ceil(x.content.length / 4), 0)
    // system 25 + question ~10 + 6 turns of ~100 = 635; room for about 3 turns.
    const fit = await fitInterview('You are Mara.', turns, 'Why?', countWhole, 400)
    expect(fit.tooBig).toBe(false)
    expect(fit.dropped).toBeGreaterThan(0)
    expect(fit.messages[0]).toEqual({ role: 'system', content: 'You are Mara.' })
    expect(fit.messages[1].role).toBe('user')
    expect(fit.messages.at(-1)).toEqual({ role: 'user', content: 'Why?' })
    // The newest turns are the ones kept.
    expect(fit.messages.some((m) => m.content.startsWith('turn 5'))).toBe(true)
    expect(fit.messages.some((m) => m.content.startsWith('turn 0'))).toBe(false)
    const tiny = await fitInterview('You are Mara.', turns, 'Why?', countWhole, 3)
    expect(tiny.tooBig).toBe(true)
    expect(tiny.messages).toHaveLength(2)
  })

  it('refuses voice notes for an entry with a pending change, a conflict or read-only text', () => {
    const base = { state: { blocked: null }, buffers: { 'bible/mara.md': { conflict: null, readOnly: false } }, pending: null, gen: null }
    expect(voiceNoteBlock(base, 'bible/mara.md')).toBeNull()
    expect(voiceNoteBlock({ ...base, pending: { path: 'bible/mara.md' } }, 'bible/mara.md')).toMatch(/waiting/)
    expect(voiceNoteBlock({ ...base, buffers: { 'bible/mara.md': { conflict: {}, readOnly: false } } }, 'bible/mara.md')).toMatch(/conflict/)
    expect(voiceNoteBlock({ ...base, buffers: { 'bible/mara.md': { conflict: null, readOnly: true } } }, 'bible/mara.md')).toMatch(/read-only/)
    expect(voiceNoteBlock({ ...base, state: { blocked: 'bad' } }, 'bible/mara.md')).toMatch(/read-only/)
  })
})

// ---------------------------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------------------------

const FOLDER = 'C:\\novels\\stage3'
const MARA_FILE = '---\nname: Mara\nkind: character\n---\nMara keeps the light.\n'

function projectState(over: Partial<NovelProjectState> = {}): NovelProjectState {
  return {
    folder: FOLDER,
    project: defaultProject('Test'),
    outline: outline(),
    bible: bible(),
    style: { path: 'style.md', text: '', hash: 'hs', meta: { bom: false, eol: 'lf' }, readOnly: false, journal: null },
    words: {},
    missing: [],
    orphans: [],
    blocked: null,
    state: { facts: [fact('f1')], invalid: [], problem: null, canReset: false },
    dismissed: {},
    checksProblem: null,
    ...over
  }
}

describe('stage 3 store', () => {
  type Call = { channel: string; args: unknown[] }
  let calls: Call[]
  let handlers: Record<string, (p: unknown) => void>
  let texts: Record<string, string>
  let replies: string[]
  let patchReply: (ops: StateOp[]) => { rejected: { index: number; reason: string }[] }
  let st: NovelProjectState
  let contextLength = 8192
  let onStart: (() => void) | null = null

  const loadStore = async () => {
    vi.resetModules()
    const store = await import('../src/renderer/src/stores/novel')
    const models = await import('../src/renderer/src/stores/models')
    models.useModels.setState({
      activeModelId: 'm1',
      instances: [{ instanceId: 'i1', modelId: 'm1', modelName: 'M', status: 'ready', contextLength, adapters: [] } as never]
    })
    await store.useNovel.getState().init()
    await store.useNovel.getState().openProject(FOLDER)
    return store
  }

  beforeEach(() => {
    calls = []
    handlers = {}
    replies = []
    contextLength = 8192
    onStart = null
    st = projectState()
    texts = { 'manuscript/s1.md': 'Mara waited.', 'manuscript/s2.md': SCENE_TEXT, 'manuscript/s3.md': 'Oskar slept.', 'style.md': 'Plain.', 'bible/mara.md': MARA_FILE }
    patchReply = () => ({ rejected: [] })
    const invoke = async (channel: string, ...args: unknown[]): Promise<unknown> => {
      calls.push({ channel, args })
      if (channel === 'novel:openProject') return st
      if (channel === 'novel:readFile') {
        const path = args[1] as string
        return { path, text: texts[path] ?? '', hash: `h-${(texts[path] ?? '').length}`, meta: { bom: false, eol: 'lf' }, readOnly: false, journal: null }
      }
      if (channel === 'novel:saveFile') {
        const r = args[0] as { path: string; text: string }
        texts[r.path] = r.text
        return { ok: true, hash: `h-${r.text.length}` }
      }
      if (channel === 'chat:countText') return Math.ceil((args[1] as string).length / 4)
      if (channel === 'chat:countTokens') return (args[1] as { content: string }[]).reduce((n, m) => n + Math.ceil(m.content.length / 4), 0)
      if (channel === 'novel:statePatch') {
        const ops = args[1] as StateOp[]
        return { state: st, ...patchReply(ops) }
      }
      if (channel === 'novel:dismissFinding') return st
      if (channel === 'chat:start') {
        const req = args[0] as { requestId: string }
        const reply = replies.shift() ?? ''
        onStart?.()
        queueMicrotask(() => {
          handlers['chat:event']({ type: 'delta', requestId: req.requestId, content: reply })
          handlers['chat:event']({ type: 'done', requestId: req.requestId, stats: {} })
        })
        return undefined
      }
      return undefined
    }
    ;(globalThis as unknown as { window: unknown }).window = {
      rulz: { invoke, on: (ch: string, fn: (p: unknown) => void) => ((handlers[ch] = fn), () => undefined) },
      addEventListener: () => undefined,
      matchMedia: () => ({ matches: true, addEventListener: () => undefined }),
      getSelection: () => null
    }
    ;(globalThis as unknown as { localStorage: unknown }).localStorage = { getItem: () => null, setItem: () => undefined, removeItem: () => undefined }
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  const started = (): Call[] => calls.filter((c) => c.channel === 'chat:start')
  const patches = (): Call[] => calls.filter((c) => c.channel === 'novel:statePatch')

  const checkReply = JSON.stringify({ findings: [{ type: 'continuity', quote: '"I never met Oskar,"', problem: 'They met on the ferry.', conflictsWith: 'Ferry', suggestion: 'Say she barely knew him.' }, { type: 'tense', quote: 'The lamp turned.', problem: 'Tense slips.', conflictsWith: '', suggestion: '' }] })

  it('Check keeps findings with the text hash, sends the whole scene, and a dismissed finding is hidden', async () => {
    const { useNovel } = await loadStore()
    await useNovel.getState().selectScene('s2')
    replies.push(checkReply)
    await useNovel.getState().checkScene('s2')
    const entryA = useNovel.getState().findings.s2
    expect(entryA.items).toHaveLength(2)
    expect(entryA.located[findingKey(entryA.items[0])]).toBe(true)
    expect(entryA.textHash).toBeTruthy()
    // The request carries the whole scene.
    const sent = (started()[0].args[0] as { messages: { content: string }[] }).messages.map((m) => m.content).join('\n')
    expect(sent).toContain('The lamp turned.')
    // A dismissed finding is hidden; the dismissal goes through main.
    await useNovel.getState().dismissFinding('s2', entryA.items[0])
    const dismiss = calls.find((c) => c.channel === 'novel:dismissFinding')!
    expect(dismiss.args).toEqual([FOLDER, 's2', findingKey(entryA.items[0]), true])
    expect(visibleFindings(entryA.items, [findingKey(entryA.items[0])])).toEqual([entryA.items[1]])
  })

  it('Check does not dismiss when the checks file is unreadable', async () => {
    st = projectState({ checksProblem: 'checks.json is not JSON' })
    const { useNovel } = await loadStore()
    await useNovel.getState().dismissFinding('s2', { type: 'pov', quote: 'x', problem: 'p', conflictsWith: '', suggestion: '' })
    expect(calls.some((c) => c.channel === 'novel:dismissFinding')).toBe(false)
  })

  it('Check is refused, and nothing is sent, when the whole scene does not fit', async () => {
    contextLength = 2048
    texts['manuscript/s2.md'] = Array.from({ length: 300 }, (_, i) => `Paragraph ${i} with enough words to take up room in a small context.`).join('\n\n')
    const { useNovel } = await loadStore()
    await useNovel.getState().checkScene('s2')
    expect(started()).toHaveLength(0)
    expect(useNovel.getState().findings.s2).toBeUndefined()
    expect(useNovel.getState().gen).toBeNull()
  })

  it('Update state: card ties to the text, Apply sends one patch from/until this scene, rejected reasons are kept', async () => {
    const { useNovel } = await loadStore()
    await useNovel.getState().selectScene('s2')
    replies.push(
      JSON.stringify({
        changes: [
          { op: 'add', entity: 'Mara', kind: 'carries', value: 'the brass key', about: null },
          { op: 'add', entity: 'Ghost', kind: 'knows', value: 'x', about: null },
          { op: 'end', factId: 'f1', reason: 'she left' },
          { op: 'end', factId: 'nope', reason: '' }
        ]
      })
    )
    await useNovel.getState().updateState('s2')
    const card = useNovel.getState().stateProposal!
    expect(card.status).toBe('ready')
    expect(card.items).toHaveLength(2)
    expect(card.left).toHaveLength(2)
    expect(patches()).toHaveLength(0)
    patchReply = () => ({ rejected: [{ index: 1, reason: 'f1 already has an end' }] })
    await useNovel.getState().applyStateProposal()
    expect(patches()).toHaveLength(1)
    const ops = patches()[0].args[1] as StateOp[]
    expect(ops).toHaveLength(2)
    expect(ops[0]).toMatchObject({ op: 'add', fact: { entity: 'mara', kind: 'carries', value: 'the brass key', from: 's2', until: null } })
    expect((ops[0] as { fact: { id: string } }).fact.id).toMatch(/^f/)
    expect(ops[1]).toEqual({ op: 'end', factId: 'f1', until: 's2' })
    const after = useNovel.getState().stateProposal!
    expect(after.status).toBe('applied')
    expect(after.applied).toBe(1)
    expect(after.rejected[0].reason).toBe('f1 already has an end')
    expect(after.rejected[0].label).toContain('Ends')
  })

  it('Update state: Apply is refused after the scene text changed', async () => {
    const { useNovel } = await loadStore()
    await useNovel.getState().selectScene('s2')
    replies.push(JSON.stringify({ changes: [{ op: 'add', entity: 'Mara', kind: 'carries', value: 'a key', about: null }] }))
    await useNovel.getState().updateState('s2')
    expect(useNovel.getState().stateProposal?.status).toBe('ready')
    useNovel.getState().edit('manuscript/s2.md', `${SCENE_TEXT} More.`)
    await useNovel.getState().applyStateProposal()
    expect(patches()).toHaveLength(0)
    expect(useNovel.getState().stateProposal?.refused).toMatch(/changed/)
  })

  it('Update state: an unreadable reply shows the raw text and changes nothing', async () => {
    const { useNovel } = await loadStore()
    await useNovel.getState().selectScene('s2')
    replies.push('not json at all')
    await useNovel.getState().updateState('s2')
    const card = useNovel.getState().stateProposal!
    expect(card.status).toBe('invalid')
    expect(card.raw).toBe('not json at all')
    expect(patches()).toHaveLength(0)
  })

  it('State tab ops go to novel:statePatch as given, and nothing is sent for a read-only state', async () => {
    const { useNovel } = await loadStore()
    const ops: StateOp[] = [{ op: 'end', factId: 'f1', until: 's3' }, { op: 'remove', factId: 'f1' }]
    patchReply = () => ({ rejected: [{ index: 1, reason: 'no such fact' }] })
    const r = await useNovel.getState().statePatch(ops)
    expect(patches()[0].args).toEqual([FOLDER, ops])
    expect(r?.rejected).toEqual([{ index: 1, reason: 'no such fact' }])
    st = projectState({ state: { facts: [], invalid: [], problem: 'state.json is not JSON', canReset: false } })
    useNovel.setState({ state: st })
    expect(await useNovel.getState().statePatch([{ op: 'remove', factId: 'f1' }])).toBeNull()
    expect(patches()).toHaveLength(1)
    // An empty file can be reset, and only that.
    st = projectState({ state: { facts: [], invalid: [], problem: 'state.json is empty', canReset: true } })
    useNovel.setState({ state: st })
    await useNovel.getState().statePatch([{ op: 'reset' }])
    expect(patches()).toHaveLength(2)
  })

  it('Interview: history per character and scene, own request, scene buffers untouched', async () => {
    const { useNovel } = await loadStore()
    await useNovel.getState().selectScene('s2')
    const before = useNovel.getState().buffers['manuscript/s2.md']
    replies.push('At the lighthouse, as always.')
    expect(await useNovel.getState().askInterview('mara', 'Where were you?')).toBe(true)
    const req = started()[0].args[0] as { messages: { role: string; content: string }[]; requestId: string }
    expect(req.messages[0].role).toBe('system')
    expect(req.messages[0].content).toContain('You are Mara.')
    expect(req.messages[0].content).toContain('location: the lighthouse')
    expect(req.messages.at(-1)).toEqual({ role: 'user', content: 'Where were you?' })
    const key = interviewKey('mara', 's2')
    expect(useNovel.getState().interviews[key]).toEqual([
      { role: 'user', content: 'Where were you?' },
      { role: 'assistant', content: 'At the lighthouse, as always.' }
    ])
    // Not the scene's buffer, not a pending change.
    expect(useNovel.getState().buffers['manuscript/s2.md']).toBe(before)
    expect(useNovel.getState().pending).toBeNull()
    // A second question carries the first exchange; another scene or character has its own history.
    replies.push('Fine.')
    await useNovel.getState().askInterview('mara', 'And then?')
    expect((started()[1].args[0] as { messages: unknown[] }).messages).toHaveLength(4)
    await useNovel.getState().selectScene('s3')
    replies.push('Hm.')
    await useNovel.getState().askInterview('oskar', 'Tired?')
    expect(Object.keys(useNovel.getState().interviews).sort()).toEqual([interviewKey('mara', 's2'), interviewKey('oskar', 's3')].sort())
    expect((started()[2].args[0] as { messages: unknown[] }).messages).toHaveLength(2)
    useNovel.getState().clearInterview(key)
    expect(useNovel.getState().interviews[key]).toBeUndefined()
  })

  it('Add to voice notes edits the entry buffer, and is refused while the entry has a pending change', async () => {
    const { useNovel } = await loadStore()
    expect(await useNovel.getState().addVoiceNote('mara', 'She never swears.')).toBe(true)
    const buf = useNovel.getState().buffers['bible/mara.md']
    expect(buf.text).toContain('## Voice')
    expect(buf.text).toContain('She never swears.')
    expect(buf.text.startsWith('---\nname: Mara\nkind: character\n---\n')).toBe(true)
    expect(buf.dirty).toBe(true)

    const before = buf.text
    useNovel.setState({
      pending: { id: 'p', path: 'bible/mara.md', sceneId: 's1', action: 'rewrite', label: 'x', baseText: before, baseHash: buf.baseHash, version: buf.version, alts: [], active: 0, newSticky: [], again: { action: 'rewrite', note: '', selection: null, cursor: 0, count: 1 } }
    })
    expect(await useNovel.getState().addVoiceNote('mara', 'Another note.')).toBe(false)
    expect(useNovel.getState().buffers['bible/mara.md'].text).toBe(before)
    useNovel.setState({ pending: null })
    // A conflict blocks it too.
    useNovel.setState((s) => ({ buffers: { ...s.buffers, 'bible/mara.md': { ...s.buffers['bible/mara.md'], conflict: { diskHash: 'x', diskText: 'y' } } } }))
    expect(await useNovel.getState().addVoiceNote('mara', 'Another note.')).toBe(false)
  })

  it('Apply sends nothing when the project is switched while the scene is read, and locks the card meanwhile', async () => {
    const { useNovel } = await loadStore()
    await useNovel.getState().selectScene('s2')
    replies.push(JSON.stringify({ changes: [{ op: 'add', entity: 'Mara', kind: 'carries', value: 'a key', about: null }] }))
    await useNovel.getState().updateState('s2')
    const first = useNovel.getState().applyStateProposal()
    expect(useNovel.getState().stateProposal?.applying).toBe(true)
    // A second click while the first is reading does nothing.
    const second = useNovel.getState().applyStateProposal()
    useNovel.setState({ folder: 'C:\\novels\\other' })
    await Promise.all([first, second])
    expect(patches()).toHaveLength(0)
  })

  it('Stop during preparation sends nothing (Check and the interview)', async () => {
    const { useNovel } = await loadStore()
    await useNovel.getState().selectScene('s2')
    replies.push(checkReply)
    const check = useNovel.getState().checkScene('s2')
    useNovel.getState().stop()
    await check
    expect(started()).toHaveLength(0)
    expect(useNovel.getState().findings.s2).toBeUndefined()
    expect(useNovel.getState().gen).toBeNull()
    replies.push('Hi.')
    const ask = useNovel.getState().askInterview('mara', 'Hello?')
    useNovel.getState().stop()
    expect(await ask).toBe(false)
    expect(started()).toHaveLength(0)
    expect(useNovel.getState().interviewRun).toBeNull()
  })

  it('the interview run records its key while it streams, and clears it after', async () => {
    const { useNovel } = await loadStore()
    await useNovel.getState().selectScene('s2')
    let seen: string | null = null
    onStart = () => (seen = useNovel.getState().interviewRun?.key ?? null)
    replies.push('Hi.')
    await useNovel.getState().askInterview('mara', 'Hello?')
    expect(seen).toBe(interviewKey('mara', 's2'))
    expect(useNovel.getState().interviewRun).toBeNull()
  })

  it('Check drops its result when the scene text moved while the request ran', async () => {
    const { useNovel } = await loadStore()
    await useNovel.getState().selectScene('s2')
    onStart = () =>
      useNovel.setState((x) => ({ buffers: { ...x.buffers, 'manuscript/s2.md': { ...x.buffers['manuscript/s2.md'], text: 'Something else entirely.', version: x.buffers['manuscript/s2.md'].version + 1 } } }))
    replies.push(checkReply)
    await useNovel.getState().checkScene('s2')
    expect(useNovel.getState().findings.s2).toBeUndefined()
  })

  it('Add to voice notes does nothing when the project changed while the entry was read', async () => {
    const { useNovel } = await loadStore()
    const add = useNovel.getState().addVoiceNote('mara', 'A note.')
    useNovel.setState({ folder: 'C:\\novels\\other' })
    expect(await add).toBe(false)
    expect(useNovel.getState().buffers['bible/mara.md']?.text ?? MARA_FILE).toBe(MARA_FILE)
  })

  it('closing the project clears findings, the card and the interviews', async () => {
    const { useNovel } = await loadStore()
    await useNovel.getState().selectScene('s2')
    replies.push(checkReply)
    await useNovel.getState().checkScene('s2')
    replies.push('Hi.')
    await useNovel.getState().askInterview('mara', 'Hello?')
    expect(await useNovel.getState().closeProject()).toBe(true)
    const s = useNovel.getState()
    expect(s.findings).toEqual({})
    expect(s.interviews).toEqual({})
    expect(s.stateProposal).toBeNull()
  })
})
