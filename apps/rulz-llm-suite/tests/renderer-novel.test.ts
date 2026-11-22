/// <reference lib="dom" />
// The store and run library are renderer code; the DOM types let the node project check them.
declare global {
  interface Window {
    rulz: import('@shared/ipc').RulzApi
  }
}

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { BibleEntry, NovelOutline, NovelProjectState, NovelTextFile } from '@shared/novel'
import { defaultProject } from '@shared/novelPrompts'
import { sceneHashKey } from '@shared/novelText'
import {
  applyResult,
  buildRequest,
  collectSummaries,
  doesNotFit,
  fitRequest,
  hashText,
  pendingStale,
  resolveSceneRefs,
  seedsFor,
  summaryHealth,
  targetWordsFor,
  type RunContext
} from '../src/renderer/src/lib/novelRun'

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
  status: 'beats' as const,
  pov: null,
  present: [] as string[],
  sticky: [] as string[],
  targetWords: 2500 as number | null,
  ...extra
})

function outline(): NovelOutline {
  return {
    format: 1,
    chapters: [
      { id: 'c1', title: 'One', act: '', summary: '', summaryOf: '', scenes: [scene('s1', 'Ferry', { summary: 'They cross.', summaryOf: 'h1' })] },
      { id: 'c2', title: 'Two', act: '', summary: '', summaryOf: '', scenes: [scene('s2', 'Dock', { summary: 'Old summary.', summaryOf: 'old' }), scene('s3', 'Inn', { beats: 'Mara meets a witch.' })] }
    ]
  }
}

const count = async (t: string): Promise<number> => Math.ceil(t.length / 4)

function ctx(over: Partial<RunContext> = {}): RunContext {
  const text = 'First paragraph.\n\nSecond paragraph.\n\nThird one mentions Oskar.'
  return {
    action: 'continue',
    project: defaultProject('Test'),
    style: 'Dry and plain.',
    outline: outline(),
    bible: [entry('mara', 'Mara', { always: true }), entry('oskar', 'Oskar'), entry('witch', 'Hedda', { triggers: ['witch'] })],
    sceneId: 's3',
    sceneHashes: { 'manuscript/s1.md': 'h1', 'manuscript/s2.md': 'new', 'manuscript/s3.md': 'h3' },
    text,
    cursor: text.length,
    selection: null,
    note: '',
    contextLength: 8192,
    templateOverhead: 20,
    ...over
  }
}

describe('collectSummaries', () => {
  it('uses current summaries and falls back to beats for stale or missing ones', () => {
    const o = outline()
    const items = collectSummaries(o, { 'manuscript/s1.md': 'h1', 'manuscript/s2.md': 'new', 'manuscript/s3.md': 'h3' }, 1, 1)
    // Chapter 1 has no chapter summary: its plan comes from the scene summaries.
    expect(items[0]).toMatchObject({ kind: 'chapter', fallback: 'missing' })
    expect(items[0].content).toContain('They cross.')
    // Scene "Dock" has a summary of another version of the file: its beats are used.
    expect(items[1]).toMatchObject({ kind: 'scene', fallback: 'stale', content: 'Beats of Dock' })
  })

  it('trusts a chapter summary whose scene hashes match', () => {
    const o = outline()
    o.chapters[0].summary = 'Chapter one happened.'
    o.chapters[0].summaryOf = sceneHashKey(['h1'])
    const items = collectSummaries(o, { 'manuscript/s1.md': 'h1' }, 1, null)
    expect(items).toEqual([{ label: 'Chapter 1: One', kind: 'chapter', content: 'Chapter one happened.', fallback: 'none' }])
  })

  it('counts written scenes with missing or stale summaries', () => {
    const h = summaryHealth(outline(), { 'manuscript/s1.md': 'h1', 'manuscript/s2.md': 'new', 'manuscript/s3.md': 'h3' }, { 'manuscript/s1.md': 100, 'manuscript/s2.md': 50, 'manuscript/s3.md': 10 })
    expect(h).toEqual({ missing: ['s3'], stale: ['s2'] })
  })
})

describe('buildRequest', () => {
  it('assembles a Continue request with triggered entries, fixed entries and summaries', async () => {
    const built = await buildRequest(ctx({ note: 'a witch appears' }), count)
    expect(built.result.messages[0].role).toBe('system')
    const user = built.result.messages[1].content
    expect(user).toContain('Third one mentions Oskar.')
    expect(user).toContain('Mara')
    expect(built.candidates.mara).toMatchObject({ fixed: true, reason: 'always' })
    expect(built.candidates.oskar.reason).toBe('trigger: Oskar')
    expect(built.candidates.witch.reason).toBe('trigger: witch')
    expect(built.newSticky.sort()).toEqual(['oskar', 'witch'])
    expect(built.fallbacks).toEqual({ missing: 1, stale: 1 })
    expect(built.result.includedEntries).toContain('mara')
  })

  it('does not offer sticky entries again and keeps them in the request', async () => {
    const o = outline()
    o.chapters[1].scenes[1].sticky = ['witch']
    const built = await buildRequest(ctx({ outline: o }), count)
    expect(built.candidates.witch.reason).toBe('sticky')
    expect(built.newSticky).not.toContain('witch')
  })

  it('sizes the Continue task to the reply budget, not the scene target', async () => {
    expect(targetWordsFor('continue', 2500, 1024)).toBe(600)
    expect(targetWordsFor('draftScene', 2500, 1024)).toBe(2500)
    expect(targetWordsFor('rewrite', 2500, 1024)).toBeNull()
    const built = await buildRequest(ctx(), count)
    expect(built.input.task.text).toContain('about 600 words')
    expect(built.input.task.text).not.toContain('2500')
  })

  it('puts the selection in the task and only the text before it in the prose', async () => {
    const text = 'Alpha beta.\n\nGamma delta.\n\nEpsilon.'
    const from = text.indexOf('Gamma')
    const built = await buildRequest(ctx({ action: 'rewrite', text, cursor: from, selection: { from, to: from + 12 }, variant: 'tighter' }), count)
    expect(built.input.task.text).toContain('Gamma delta.')
    expect(built.input.proseBlocks.map((b) => b.text)).toEqual(['Alpha beta.'])
    expect(built.input.proseAfter.text).toContain('Epsilon.')
  })

  it('uses the previous scene as the prose of a draft on an empty scene', async () => {
    const built = await buildRequest(ctx({ action: 'draftScene', text: '', cursor: 0, previousText: 'Last scene ends here.\n\nFinal line.' }), count)
    expect(built.input.proseBlocks.map((b) => b.text)).toEqual(['Last scene ends here.', 'Final line.'])
  })

  it('throws a plain message when the scene is gone', async () => {
    await expect(buildRequest(ctx({ sceneId: 'nope' }), count)).rejects.toThrow('not in the outline')
  })

  it('keeps the anchor step independent of the text, so the prose start is stable', async () => {
    const a = await buildRequest(ctx(), count)
    const b = await buildRequest(ctx({ text: 'x\n\n'.repeat(500), cursor: 1500 }), count)
    expect(a.input.anchorStep).toBe(b.input.anchorStep)
  })
})

describe('fitRequest', () => {
  it('sends when the whole request fits', async () => {
    const out = await fitRequest(ctx(), count, async () => 500)
    expect(out.tooBig).toBe(false)
    expect(out.retried).toBe(false)
  })

  it('assembles once more with the drift added to the overhead when the template count is over', async () => {
    const text = Array.from({ length: 600 }, () => 'word word word.').join('\n\n')
    const long = ctx({ text, cursor: text.length, contextLength: 4096, templateOverhead: 20 })
    let calls = 0
    const out = await fitRequest(long, count, async () => (calls++ === 0 ? 4096 + 400 : 1000))
    expect(out.retried).toBe(true)
    expect(out.built.input.templateOverhead).toBeGreaterThan(20)
    expect(out.tooBig).toBe(false)
  })

  it('refuses a request whose fixed parts are larger than the context', async () => {
    const out = await fitRequest(ctx({ style: 'x'.repeat(40000), contextLength: 2048 }), count, async () => null)
    expect(out.tooBig).toBe(true)
    expect(doesNotFit(out.built)).toBe(true)
  })
})

describe('results and pending changes', () => {
  it('inserts Continue text a blank line from its neighbours', () => {
    expect(applyResult('continue', 'One.', 4, null, ' Two. ')).toBe('One.\n\nTwo.')
    expect(applyResult('continue', 'One.\n\n', 6, null, 'Two.')).toBe('One.\n\nTwo.')
    expect(applyResult('continue', 'One.\n\nThree.', 4, null, 'Two.')).toBe('One.\n\nTwo.\n\n\n\nThree.'.replace('\n\n\n\n', '\n\n'))
  })

  it('replaces the selection for rewrite and takes the whole text for a draft', () => {
    expect(applyResult('rewrite', 'a big dog ran', 0, { from: 2, to: 5 }, 'huge')).toBe('a huge dog ran')
    expect(applyResult('draftScene', '', 0, null, '\nThe scene.\n')).toBe('The scene.')
  })

  it('refuses a pending change when the hash or the version moved', () => {
    const p = { baseHash: 'a', version: 3 }
    expect(pendingStale(p, { baseHash: 'a', version: 3 })).toBe(false)
    expect(pendingStale(p, { baseHash: 'b', version: 3 })).toBe(true)
    expect(pendingStale(p, { baseHash: 'a', version: 4 })).toBe(true)
  })

  it('hashes text stably and gives different alternatives different seeds', () => {
    expect(hashText('abc')).toBe(hashText('abc'))
    expect(hashText('abc')).not.toBe(hashText('abd'))
    expect(seedsFor(10, 3)).toEqual([10, 11, 12])
    expect(new Set(seedsFor(null, 4)).size).toBe(4)
  })
})

// ---------------------------------------------------------------------------------------------
// Store: autosave, journal, failed saves, conflicts and the flush on close
// ---------------------------------------------------------------------------------------------

const PATH = 'manuscript/s1.md'

function projectState(): NovelProjectState {
  const o = outline()
  o.chapters = [o.chapters[0]]
  return {
    folder: 'C:\\novels\\test',
    project: defaultProject('Test'),
    outline: o,
    bible: [],
    style: { path: 'style.md', text: '', hash: 'hs', meta: { bom: false, eol: 'lf' }, readOnly: false, journal: null },
    words: { [PATH]: 3 },
    missing: [],
    orphans: [],
    blocked: null,
    state: { facts: [], invalid: [], problem: null, canReset: false },
    dismissed: {},
    checksProblem: null
  }
}

function file(text: string, hash: string, journal: NovelTextFile['journal'] = null): NovelTextFile {
  return { path: PATH, text, hash, meta: { bom: false, eol: 'lf' }, readOnly: false, journal }
}

describe('novel store', () => {
  type Call = { channel: string; args: unknown[] }
  let calls: Call[]
  let saveReply: () => unknown
  let diskText: string
  let diskHash: string

  const loadStore = async () => {
    vi.resetModules()
    const store = await import('../src/renderer/src/stores/novel')
    await store.useNovel.getState().openProject('C:\\novels\\test')
    return store
  }

  beforeEach(async () => {
    vi.useFakeTimers()
    calls = []
    diskText = 'Opening line.'
    diskHash = 'h1'
    saveReply = () => ({ ok: true, hash: 'h2' })
    const invoke = async (channel: string, ...args: unknown[]): Promise<unknown> => {
      calls.push({ channel, args })
      if (channel === 'novel:openProject') return projectState()
      if (channel === 'novel:readFile') return file(diskText, diskHash)
      if (channel === 'novel:saveFile') return saveReply()
      return undefined
    }
    ;(globalThis as unknown as { window: unknown }).window = {
      rulz: { invoke, on: () => () => undefined },
      addEventListener: () => undefined,
      matchMedia: () => ({ matches: true, addEventListener: () => undefined })
    }
    ;(globalThis as unknown as { localStorage: unknown }).localStorage = { getItem: () => null, setItem: () => undefined, removeItem: () => undefined }
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  const count = (channel: string): number => calls.filter((c) => c.channel === channel).length

  it('opens the first scene and journals 300 ms after an edit, then saves after 1 s', async () => {
    const { useNovel } = await loadStore()
    expect(useNovel.getState().buffers[PATH].text).toBe('Opening line.')
    useNovel.getState().edit(PATH, 'Opening line. More.')
    expect(useNovel.getState().buffers[PATH].dirty).toBe(true)
    await vi.advanceTimersByTimeAsync(299)
    expect(count('novel:journal')).toBe(0)
    await vi.advanceTimersByTimeAsync(2)
    expect(calls.find((c) => c.channel === 'novel:journal')?.args).toEqual(['C:\\novels\\test', PATH, 'Opening line. More.', 'h1'])
    expect(count('novel:saveFile')).toBe(0)
    await vi.advanceTimersByTimeAsync(800)
    const save = calls.find((c) => c.channel === 'novel:saveFile')!.args[0] as { text: string; baseHash: string }
    expect(save).toMatchObject({ path: PATH, text: 'Opening line. More.', baseHash: 'h1' })
    const b = useNovel.getState().buffers[PATH]
    expect(b).toMatchObject({ dirty: false, baseHash: 'h2', error: null })
  })

  it('keeps the text when a save fails and tries again every 5 seconds', async () => {
    const { useNovel } = await loadStore()
    saveReply = () => ({ ok: false, reason: 'error', error: 'file is locked' })
    useNovel.getState().edit(PATH, 'Changed.')
    await vi.advanceTimersByTimeAsync(1100)
    expect(useNovel.getState().buffers[PATH]).toMatchObject({ dirty: true, error: 'file is locked', text: 'Changed.' })
    expect(count('novel:saveFile')).toBe(1)
    saveReply = () => ({ ok: true, hash: 'h3' })
    await vi.advanceTimersByTimeAsync(5100)
    expect(count('novel:saveFile')).toBe(2)
    expect(useNovel.getState().buffers[PATH]).toMatchObject({ dirty: false, error: null, baseHash: 'h3' })
  })

  it('stops autosaving after a conflict and keeps the text', async () => {
    const { useNovel } = await loadStore()
    saveReply = () => ({ ok: false, reason: 'conflict', diskHash: 'hx', diskText: 'Someone else wrote this.' })
    useNovel.getState().edit(PATH, 'Mine.')
    await vi.advanceTimersByTimeAsync(1100)
    expect(useNovel.getState().buffers[PATH].conflict).toEqual({ diskHash: 'hx', diskText: 'Someone else wrote this.' })
    useNovel.getState().edit(PATH, 'Mine, more.')
    await vi.advanceTimersByTimeAsync(3000)
    expect(count('novel:saveFile')).toBe(1)
    expect(useNovel.getState().buffers[PATH].text).toBe('Mine, more.')
  })

  it('flushAll journals and saves every dirty buffer', async () => {
    const { useNovel, flushAll } = await loadStore()
    useNovel.getState().edit(PATH, 'Unsaved at close.')
    const done = flushAll()
    await vi.advanceTimersByTimeAsync(10)
    await done
    expect(calls.some((c) => c.channel === 'novel:journal')).toBe(true)
    expect((calls.find((c) => c.channel === 'novel:saveFile')!.args[0] as { text: string }).text).toBe('Unsaved at close.')
    expect(useNovel.getState().buffers[PATH].dirty).toBe(false)
  })

  it('does not take text while a change is pending, and refuses a stale Accept', async () => {
    const { useNovel } = await loadStore()
    const b = useNovel.getState().buffers[PATH]
    useNovel.setState({
      pending: {
        id: 'p1',
        path: PATH,
        sceneId: 's1',
        action: 'continue',
        label: 'Continue',
        baseText: b.text,
        baseHash: 'old-hash',
        version: b.version,
        alts: [{ text: `${b.text}\n\nNew.`, generated: 'New.', seed: 1 }],
        active: 0,
        newSticky: [],
        again: { action: 'continue', note: '', selection: null, cursor: b.text.length, count: 1 }
      }
    })
    useNovel.getState().edit(PATH, 'typed while pending')
    expect(useNovel.getState().buffers[PATH].text).toBe('Opening line.')
    await useNovel.getState().resolvePending(`${b.text}\n\nNew.`)
    expect(useNovel.getState().pendingRefused).toBe(true)
    expect(useNovel.getState().pending).not.toBeNull()
    expect(count('novel:saveFile')).toBe(0)
  })

  it('accepts a pending change against the same base: edits the buffer and saves with a snapshot', async () => {
    const { useNovel } = await loadStore()
    const b = useNovel.getState().buffers[PATH]
    useNovel.setState({
      pending: {
        id: 'p2',
        path: PATH,
        sceneId: 's1',
        action: 'continue',
        label: 'Continue',
        baseText: b.text,
        baseHash: b.baseHash,
        version: b.version,
        alts: [{ text: `${b.text}\n\nNew.`, generated: 'New.', seed: 1 }],
        active: 0,
        newSticky: [],
        again: { action: 'continue', note: '', selection: null, cursor: b.text.length, count: 1 }
      }
    })
    await useNovel.getState().resolvePending(`${b.text}\n\nNew.`)
    const req = calls.find((c) => c.channel === 'novel:saveFile')!.args[0] as { text: string; snapshot?: boolean }
    expect(req).toMatchObject({ text: 'Opening line.\n\nNew.', snapshot: true })
    expect(useNovel.getState().pending).toBeNull()
  })

  it('rejecting everything leaves the buffer untouched and writes nothing', async () => {
    const { useNovel } = await loadStore()
    const b = useNovel.getState().buffers[PATH]
    useNovel.setState({
      pending: {
        id: 'p3',
        path: PATH,
        sceneId: 's1',
        action: 'continue',
        label: 'Continue',
        baseText: b.text,
        baseHash: b.baseHash,
        version: b.version,
        alts: [{ text: `${b.text}\n\nNew.`, generated: 'New.', seed: 1 }],
        active: 0,
        newSticky: [],
        again: { action: 'continue', note: '', selection: null, cursor: 0, count: 1 }
      }
    })
    await useNovel.getState().resolvePending(b.text)
    expect(useNovel.getState().pending).toBeNull()
    expect(useNovel.getState().buffers[PATH].text).toBe('Opening line.')
    expect(count('novel:saveFile')).toBe(0)
  })
})

// ---------------------------------------------------------------------------------------------
// Store: review fixes (open the same folder, in-flight saves, bounded flush, reload, style, runs)
// ---------------------------------------------------------------------------------------------

describe('novel store: data safety fixes', () => {
  type Call = { channel: string; args: unknown[] }
  let calls: Call[]
  let handlers: Record<string, (payload: unknown) => void>
  let gate: { save: Promise<unknown> | null; journal: Promise<unknown> | null; read: Promise<unknown> | null }
  let texts: Record<string, { text: string; hash: string }>
  let saveHashes: number

  const loadStore = async () => {
    vi.resetModules()
    const store = await import('../src/renderer/src/stores/novel')
    await store.useNovel.getState().init()
    await store.useNovel.getState().openProject('C:\\novels\\test')
    return store
  }

  beforeEach(() => {
    vi.useFakeTimers()
    calls = []
    handlers = {}
    gate = { save: null, journal: null, read: null }
    texts = { [PATH]: { text: 'A', hash: 'h1' }, 'style.md': { text: 'Style.', hash: 'hs' } }
    saveHashes = 1
    const invoke = async (channel: string, ...args: unknown[]): Promise<unknown> => {
      calls.push({ channel, args })
      if (channel === 'novel:openProject') return projectState()
      if (channel === 'novel:readFile') {
        const path = args[1] as string
        const t = texts[path] ?? { text: '', hash: '' }
        if (gate.read && path === PATH) await gate.read
        return { path, text: t.text, hash: t.hash, meta: { bom: false, eol: 'lf' }, readOnly: false, journal: null }
      }
      if (channel === 'novel:saveFile') {
        if (gate.save) await gate.save
        return { ok: true, hash: `hs${++saveHashes}` }
      }
      if (channel === 'novel:journal') {
        if (gate.journal) await gate.journal
        return undefined
      }
      if (channel === 'chat:countText') return 5
      if (channel === 'chat:countTokens') return 50
      return undefined
    }
    ;(globalThis as unknown as { window: unknown }).window = {
      rulz: { invoke, on: (ch: string, fn: (p: unknown) => void) => ((handlers[ch] = fn), () => undefined) },
      addEventListener: () => undefined,
      matchMedia: () => ({ matches: true, addEventListener: () => undefined })
    }
    ;(globalThis as unknown as { localStorage: unknown }).localStorage = { getItem: () => null, setItem: () => undefined, removeItem: () => undefined }
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  const count = (channel: string): number => calls.filter((c) => c.channel === channel).length

  it('opening the folder that is already open leaves buffers and timers alone (finding 1)', async () => {
    const { useNovel } = await loadStore()
    useNovel.getState().edit(PATH, 'A edited')
    await useNovel.getState().openProject('C:\\novels\\test')
    expect(useNovel.getState().buffers[PATH].text).toBe('A edited')
    expect(useNovel.getState().buffers[PATH].dirty).toBe(true)
    await vi.advanceTimersByTimeAsync(1100)
    expect((calls.find((c) => c.channel === 'novel:saveFile')!.args[0] as { text: string }).text).toBe('A edited')
  })

  it('undoing to the old text while a save is in flight is not clean (finding 2)', async () => {
    const { useNovel } = await loadStore()
    let release: (v?: unknown) => void = () => undefined
    gate.save = new Promise((r) => (release = r))
    useNovel.getState().edit(PATH, 'B')
    await vi.advanceTimersByTimeAsync(1100)
    expect(count('novel:saveFile')).toBe(1)
    useNovel.getState().edit(PATH, 'A')
    expect(useNovel.getState().buffers[PATH].dirty).toBe(true)
    gate.save = null
    release()
    await vi.advanceTimersByTimeAsync(10)
    // The disk got B; the editor holds A, so it is still unsaved and gets saved again.
    expect(useNovel.getState().buffers[PATH]).toMatchObject({ savedText: 'B', text: 'A', dirty: true })
    await vi.advanceTimersByTimeAsync(1100)
    const saves = calls.filter((c) => c.channel === 'novel:saveFile').map((c) => (c.args[0] as { text: string }).text)
    expect(saves).toEqual(['B', 'A'])
    expect(useNovel.getState().buffers[PATH].dirty).toBe(false)
  })

  it('a journal that never answers does not hang the flush, and the project stays open (finding 3)', async () => {
    const { useNovel, flushAll, FLUSH_BUDGET_MS } = await loadStore()
    gate.journal = new Promise(() => undefined)
    gate.save = new Promise(() => undefined)
    useNovel.getState().edit(PATH, 'Precious text');
    let result: { ok: boolean; failed: string[] } | null = null
    void flushAll().then((r) => (result = r))
    await vi.advanceTimersByTimeAsync(FLUSH_BUDGET_MS + 100)
    expect(result).toEqual({ ok: false, failed: [PATH] })
    const closed = useNovel.getState().closeProject()
    await vi.advanceTimersByTimeAsync(FLUSH_BUDGET_MS + 100)
    expect(await closed).toBe(false)
    expect(useNovel.getState().folder).toBe('C:\\novels\\test')
    expect(useNovel.getState().buffers[PATH].text).toBe('Precious text')
  })

  it('text typed during the flush is saved, not lost (finding 3)', async () => {
    const { useNovel, flushAll } = await loadStore()
    let release: (v?: unknown) => void = () => undefined
    gate.journal = new Promise((r) => (release = r))
    useNovel.getState().edit(PATH, 'one')
    let result: { ok: boolean } | null = null
    void flushAll().then((r) => (result = r))
    await vi.advanceTimersByTimeAsync(10)
    useNovel.getState().edit(PATH, 'one two')
    gate.journal = null
    release()
    await vi.advanceTimersByTimeAsync(50)
    expect(result).toEqual({ ok: true, failed: [] })
    const saved = calls.filter((c) => c.channel === 'novel:saveFile').map((c) => (c.args[0] as { text: string }).text)
    expect(saved.at(-1)).toBe('one two')
    expect(useNovel.getState().buffers[PATH].dirty).toBe(false)
  })

  it('Reload from disk does not replace text typed while the file was being read (finding 4)', async () => {
    const { useNovel } = await loadStore()
    let release: (v?: unknown) => void = () => undefined
    gate.read = new Promise((r) => (release = r))
    texts[PATH] = { text: 'From disk', hash: 'hd' }
    const reload = useNovel.getState().reloadFromDisk(PATH)
    useNovel.getState().edit(PATH, 'typed meanwhile')
    release()
    await reload
    expect(useNovel.getState().buffers[PATH].text).toBe('typed meanwhile')
  })

  it('a state change to style.md marks a dirty style buffer instead of replacing it, and reloads a clean one (finding 5)', async () => {
    const { useNovel } = await loadStore()
    const st = projectState()
    texts['style.md'] = { text: 'New style.', hash: 'hs2' }
    st.style = { ...st.style, text: 'New style.', hash: 'hs2' }
    useNovel.getState().edit('style.md', 'My style edit')
    handlers['novel:changed']({ kind: 'state', folder: st.folder, state: st })
    await vi.advanceTimersByTimeAsync(5)
    expect(useNovel.getState().buffers['style.md']).toMatchObject({ text: 'My style edit', changedOnDisk: true })

    // A clean style buffer is reloaded, but not while a request is running.
    await useNovel.getState().reloadFromDisk('style.md')
    texts['style.md'] = { text: 'Newer style.', hash: 'hs3' }
    st.style = { ...st.style, text: 'Newer style.', hash: 'hs3' }
    useNovel.setState({ gen: { path: PATH, action: 'continue', label: 'Continue', phase: 'writing', alt: 1, of: 1, progress: null, text: '' } })
    handlers['novel:changed']({ kind: 'state', folder: st.folder, state: st })
    await vi.advanceTimersByTimeAsync(5)
    expect(useNovel.getState().buffers['style.md'].text).toBe('New style.')
    expect(useNovel.getState().buffers['style.md'].changedOnDisk).toBe(true)
    useNovel.setState({ gen: null })
    await useNovel.getState().reloadFromDisk('style.md')
    expect(useNovel.getState().buffers['style.md'].text).toBe('Newer style.')
  })

  it('the accepted text is journalled at once, even when its save waits (finding 7)', async () => {
    const { useNovel } = await loadStore()
    const b = useNovel.getState().buffers[PATH]
    gate.save = new Promise(() => undefined)
    useNovel.setState({
      pending: {
        id: 'p9',
        path: PATH,
        sceneId: 's1',
        action: 'continue',
        label: 'Continue',
        baseText: b.text,
        baseHash: b.baseHash,
        version: b.version,
        alts: [{ text: 'A and more', generated: 'and more', seed: 1 }],
        active: 0,
        newSticky: [],
        again: { action: 'continue', note: '', selection: null, cursor: 1, count: 1 }
      }
    })
    void useNovel.getState().resolvePending('A and more')
    await vi.advanceTimersByTimeAsync(5)
    const j = calls.find((c) => c.channel === 'novel:journal')
    expect(j?.args[2]).toBe('A and more')
  })

  it('a run is dropped when the project changes while it is being prepared (finding 6)', async () => {
    const store = await loadStore()
    const { useNovel, registerEditor } = store
    const models = await import('../src/renderer/src/stores/models')
    models.useModels.setState({
      activeModelId: 'm1',
      instances: [{ instanceId: 'i1', modelId: 'm1', modelName: 'M', status: 'ready', contextLength: 8192, adapters: [] } as never]
    })
    registerEditor(PATH, { selection: () => ({ from: 1, to: 1 }) } as never)
    const started: string[] = []
    const origInvoke = window.rulz.invoke as (c: string, ...a: unknown[]) => Promise<unknown>
    ;(window.rulz as { invoke: unknown }).invoke = async (channel: string, ...args: unknown[]) => {
      if (channel === 'chat:start') {
        started.push((args[0] as { requestId: string }).requestId)
        // The user moves to another project while the request runs.
        useNovel.setState({ folder: 'C:\\novels\\other' })
        const id = (args[0] as { requestId: string }).requestId
        queueMicrotask(() => {
          handlers['chat:event']({ type: 'delta', requestId: id, content: 'New words.' })
          handlers['chat:event']({ type: 'done', requestId: id, stats: {} })
        })
        return undefined
      }
      return origInvoke(channel, ...args)
    }
    await useNovel.getState().init()
    const run = useNovel.getState().runAction({ action: 'continue' })
    await vi.advanceTimersByTimeAsync(200)
    await run
    expect(started.length).toBe(1)
    expect(useNovel.getState().pending).toBeNull()
  })

  it('closing the project stops a running request and drops its late result (finding 6)', async () => {
    const { useNovel } = await loadStore()
    useNovel.setState({ gen: { path: PATH, action: 'continue', label: 'Continue', phase: 'writing', alt: 1, of: 1, progress: null, text: '' } })
    expect(await useNovel.getState().closeProject()).toBe(true)
    expect(useNovel.getState().gen).toBeNull()
    expect(useNovel.getState().folder).toBeNull()
  })
})

describe('scene plan references', () => {
  const bible = [entry('mara-venn', 'Mara Venn', { aliases: ['the Salt Witch'] }), entry('oskar', 'Oskar')]

  it('maps ids, names and aliases, drops what matches nothing, and lists it', () => {
    const r = resolveSceneRefs(bible, [
      { pov: 'Mara Venn', present: ['oskar', 'the salt witch', 'Mara Venn', 'Ghost'] },
      { pov: 'Nobody', present: [] }
    ])
    expect(r.scenes[0]).toEqual({ pov: 'mara-venn', present: ['oskar', 'mara-venn'] })
    expect(r.scenes[1]).toEqual({ pov: null, present: [] })
    expect(r.unresolved).toEqual(['Ghost', 'Nobody'])
  })

  it('lists the entries with their ids in the Plan scenes request', async () => {
    const built = await buildRequest(ctx({ action: 'outlineScenes', sceneId: null, chapterId: 'c2', bible }), count)
    expect(built.input.scene.text).toContain('- Mara Venn (character, id: mara-venn)')
  })

  it('writes the resolved pov and present when a plan is accepted', async () => {
    vi.useFakeTimers()
    const patches: unknown[][] = []
    const invoke = async (channel: string, ...args: unknown[]): Promise<unknown> => {
      if (channel === 'novel:openProject') return { ...projectState(), bible }
      if (channel === 'novel:readFile') return file('A', 'h1')
      if (channel === 'novel:outlinePatch') {
        patches.push(args[1] as unknown[])
        return { ...projectState(), bible }
      }
      return undefined
    }
    ;(globalThis as unknown as { window: unknown }).window = {
      rulz: { invoke, on: () => () => undefined },
      addEventListener: () => undefined,
      matchMedia: () => ({ matches: true, addEventListener: () => undefined })
    }
    ;(globalThis as unknown as { localStorage: unknown }).localStorage = { getItem: () => null, setItem: () => undefined, removeItem: () => undefined }
    vi.resetModules()
    const { useNovel } = await import('../src/renderer/src/stores/novel')
    await useNovel.getState().openProject('C:\\novels\\test')
    await useNovel.getState().acceptProposal({
      id: 'x',
      kind: 'scenes',
      targetId: 'c1',
      title: 't',
      status: 'ready',
      raw: '',
      error: null,
      summaryOf: '',
      again: { count: 1, note: '' },
      scenes: { scenes: [{ title: 'Storm', beats: 'It rains.', pov: 'Mara Venn', present: ['Oskar', 'Ghost'] }] }
    })
    const ops = patches.at(-1) as { op: string; field?: string; value?: unknown }[]
    expect(ops.find((o) => o.field === 'pov')?.value).toBe('mara-venn')
    expect(ops.find((o) => o.field === 'present')?.value).toEqual(['oskar'])
    vi.useRealTimers()
  })
})
