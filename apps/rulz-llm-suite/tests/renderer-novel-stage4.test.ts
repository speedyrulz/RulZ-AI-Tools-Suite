/// <reference lib="dom" />
// Stage 4 renderer logic: recording of prose and card runs and their outcomes through novel:record,
// the region / final-text rules, and the busy check. IPC is mocked like in renderer-novel-stage3.test.ts.
declare global {
  interface Window {
    rulz: import('@shared/ipc').RulzApi
  }
}

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { toast } from 'sonner'
import type { BibleEntry, NovelOutline, NovelProjectState, RecordLine } from '@shared/novel'
import { defaultProject } from '@shared/novelPrompts'
import { classifyOutcome, finishOf, isRecordId, locateFinal, outcomeRegion, recordId, regionOf } from '../src/renderer/src/lib/novelRecord'

const busy = vi.hoisted(() => ({ reason: null as string | null }))
vi.mock('sonner', () => ({ toast: { error: vi.fn(), message: vi.fn(), success: vi.fn(), warning: vi.fn() } }))
vi.mock('../src/renderer/src/lib/aiBusy', () => ({ aiBusyReason: () => busy.reason, aiBusy: () => busy.reason !== null }))

// ---------------------------------------------------------------------------------------------
// Pure rules
// ---------------------------------------------------------------------------------------------

describe('recording rules', () => {
  it('makes 16 lowercase hex ids that differ', () => {
    const a = recordId()
    expect(isRecordId(a)).toBe(true)
    expect(a).toMatch(/^[0-9a-f]{16}$/)
    expect(recordId()).not.toBe(a)
  })

  it('classifies accepted, mixed and rejected from base, final and the alternative document', () => {
    expect(classifyOutcome('base', 'alt', 'alt')).toBe('accepted')
    expect(classifyOutcome('base', 'base', 'alt')).toBe('rejected')
    expect(classifyOutcome('base', 'half', 'alt')).toBe('mixed')
    // A change that alters nothing is not an acceptance.
    expect(classifyOutcome('same', 'same', 'same')).toBe('rejected')
  })

  it('puts the region where the text was applied: selection for rewrite, cursor plus lead for continue, all for a draft', () => {
    expect(regionOf('rewrite', 'a big dog ran', 0, { from: 2, to: 5 }, 'huge')).toEqual({ start: 2, end: 6, replaced: 'big' })
    expect(regionOf('expand', 'a big dog ran', 0, { from: 2, to: 5 }, ' huge ')).toEqual({ start: 2, end: 6, replaced: 'big' })
    // No selection: the cursor, after a blank-line lead of two, one, or none.
    expect(regionOf('continue', 'One.', 4, null, 'Two.')).toEqual({ start: 6, end: 10, replaced: '' })
    expect(regionOf('continue', 'One.\n', 5, null, 'Two.')).toEqual({ start: 6, end: 10, replaced: '' })
    expect(regionOf('dialogue', 'One.\n\n', 6, null, 'Two.')).toEqual({ start: 6, end: 10, replaced: '' })
    expect(regionOf('continue', '', 0, null, 'Two.')).toEqual({ start: 0, end: 4, replaced: '' })
    expect(regionOf('continue', 'One. Three.', 4, null, 'Two.')).toEqual({ start: 6, end: 10, replaced: '' })
    expect(regionOf('draftScene', 'old', 0, null, '\nThe scene.\n')).toEqual({ start: 0, end: 10, replaced: 'old' })
  })

  it('region of an accepted change is the alternative own; a mixed one ends at final length minus the text after it', () => {
    const geo = regionOf('rewrite', 'a big dog ran', 0, { from: 2, to: 5 }, 'huge')
    // The alternative document is 'a huge dog ran' (14); "huge" ends at 6, so 8 characters follow it.
    expect(outcomeRegion('accepted', geo, 14, 14)).toEqual({ start: 2, end: 6 })
    expect(outcomeRegion('mixed', geo, 14, 16)).toEqual({ start: 2, end: 8 })
    // Never before the start.
    expect(outcomeRegion('mixed', geo, 14, 3)).toEqual({ start: 2, end: 2 })
  })

  it('finds the region after later edits by its head and tail, and says outside otherwise', () => {
    const accepted = 'Head.\n\nGenerated text.\n\nTail.'
    const region = { start: 7, end: 22 }
    expect(locateFinal(accepted, region, accepted)).toBe('Generated text.')
    expect(locateFinal(accepted, region, 'Head.\n\nGenerated text, edited by me.\n\nTail.')).toBe('Generated text, edited by me.')
    expect(locateFinal(accepted, region, 'Head.\n\n\n\nTail.')).toBe('')
    // The head moved: outside.
    expect(locateFinal(accepted, region, 'New head.\n\nGenerated text.\n\nTail.')).toBeNull()
    // The tail was cut into the region: outside.
    expect(locateFinal(accepted, region, 'Head.\n\nGenerated te')).toBeNull()
    // Shorter than head + tail together: outside.
    expect(locateFinal(accepted, region, 'Head.')).toBeNull()
  })

  it('reads the finish of an alternative from the chat result', () => {
    expect(finishOf('stop', { stopped: false, error: null })).toBe('stop')
    expect(finishOf(undefined, { stopped: false, error: null })).toBe('stop')
    expect(finishOf('length', { stopped: false, error: null })).toBe('length')
    expect(finishOf('stopped', { stopped: true, error: null })).toBe('stopped')
    expect(finishOf('length', { stopped: true, error: null })).toBe('stopped')
    expect(finishOf(undefined, { stopped: false, error: 'boom' })).toBe('error')
  })
})

// ---------------------------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------------------------

const FOLDER = 'C:\\novels\\stage4'
const P1 = 'manuscript/s1.md'
const P2 = 'manuscript/s2.md'
const S1 = 'Mara waited.'
const GEN = 'He watched the tide come in.'

const scene = (id: string, title: string) => ({
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
  targetWords: null as number | null
})

const outline = (): NovelOutline => ({ format: 1, chapters: [{ id: 'c1', title: 'One', act: '', summary: '', summaryOf: '', scenes: [scene('s1', 'Ferry'), scene('s2', 'Dock')] }] })

function projectState(recording?: boolean): NovelProjectState {
  const project = defaultProject('Test')
  if (recording !== undefined) project.recording = recording
  return {
    folder: FOLDER,
    project,
    outline: outline(),
    bible: [] as BibleEntry[],
    style: { path: 'style.md', text: '', hash: 'hs', meta: { bom: false, eol: 'lf' }, readOnly: false, journal: null },
    words: {},
    missing: [],
    orphans: [],
    blocked: null,
    state: { facts: [], invalid: [], problem: null, canReset: false },
    dismissed: {},
    checksProblem: null
  }
}

describe('stage 4 store', () => {
  type Call = { channel: string; args: unknown[] }
  let calls: Call[]
  let handlers: Record<string, (p: unknown) => void>
  let texts: Record<string, string>
  let replies: { text: string; stopReason?: string }[]
  let st: NovelProjectState
  let recordReply: () => unknown

  const loadStore = async () => {
    vi.resetModules()
    const store = await import('../src/renderer/src/stores/novel')
    const models = await import('../src/renderer/src/stores/models')
    models.useModels.setState({
      activeModelId: 'm1',
      instances: [{ instanceId: 'i1', modelId: 'm1', modelName: 'M', status: 'ready', contextLength: 8192, adapters: [] } as never]
    })
    await store.useNovel.getState().init()
    await store.useNovel.getState().openProject(FOLDER)
    // The scene editor is a CodeMirror view in the app; the store only asks it where the cursor is.
    store.registerEditor(P1, { selection: () => ({ from: S1.length, to: S1.length }) } as never)
    store.registerEditor(P2, { selection: () => ({ from: 0, to: 0 }) } as never)
    return store
  }

  beforeEach(() => {
    vi.useFakeTimers()
    busy.reason = null
    calls = []
    handlers = {}
    replies = []
    st = projectState()
    texts = { [P1]: S1, [P2]: 'Oskar slept.', 'style.md': '' }
    recordReply = () => true
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
      if (channel === 'novel:outlinePatch') return st
      if (channel === 'app:info') return { version: '9.9.9', userData: '', platform: 'win32' }
      if (channel === 'novel:record') return recordReply()
      if (channel === 'chat:countText') return Math.ceil((args[1] as string).length / 4)
      if (channel === 'chat:countTokens') return (args[1] as { content: string }[]).reduce((n, m) => n + Math.ceil(m.content.length / 4), 0)
      if (channel === 'chat:start') {
        const req = args[0] as { requestId: string }
        const reply = replies.shift() ?? { text: '' }
        queueMicrotask(() => {
          handlers['chat:event']({ type: 'delta', requestId: req.requestId, content: reply.text })
          handlers['chat:event']({ type: 'done', requestId: req.requestId, stats: { stopReason: reply.stopReason } })
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
    vi.mocked(toast.message).mockClear()
    vi.mocked(toast.error).mockClear()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  /** Every novel:record line sent so far. */
  const lines = (): RecordLine[] => calls.filter((c) => c.channel === 'novel:record').map((c) => c.args[1] as RecordLine)
  const ofType = <T extends RecordLine['type']>(type: T): Extract<RecordLine, { type: T }>[] => lines().filter((l) => l.type === type) as Extract<RecordLine, { type: T }>[]
  const settle = async (): Promise<void> => {
    await vi.advanceTimersByTimeAsync(5)
  }

  /** Runs Continue in scene 1 with a reply and waits until the pending change is there. */
  const runContinue = async (store: Awaited<ReturnType<typeof loadStore>>, reply = { text: GEN, stopReason: 'stop' as string | undefined }) => {
    replies.push(reply)
    await store.useNovel.getState().selectScene('s1')
    await store.useNovel.getState().runAction({ action: 'continue' })
    await settle()
    return store.useNovel.getState().pending!
  }

  it('writes the request line when a run ends: model, sampling, messages, seed, generated text and finish', async () => {
    const store = await loadStore()
    const p = await runContinue(store)
    expect(p.recId).toMatch(/^[0-9a-f]{16}$/)
    const [req] = ofType('request')
    expect(ofType('request')).toHaveLength(1)
    expect(req).toMatchObject({ format: 1, type: 'request', id: p.recId, app: '9.9.9', action: 'continue', sceneId: 's1', chapterId: 'c1', model: { id: 'm1', name: 'M' }, adapters: [] })
    expect(req.alts).toEqual([{ index: 0, seed: p.alts[0].seed, generated: GEN, finish: 'stop' }])
    // The sampling and messages are what was sent to the chat path.
    const sent = calls.find((c) => c.channel === 'chat:start')!.args[0] as { messages: { role: string; content: string }[]; sampling: { seed: number; maxTokens: number; thinking: string } }
    expect(req.messages).toEqual(sent.messages.map((m) => ({ role: m.role, content: m.content })))
    expect(req.sampling).toMatchObject({ seed: sent.sampling.seed, maxTokens: sent.sampling.maxTokens, thinking: 'off' })
    expect(calls.filter((c) => c.channel === 'novel:record').every((c) => c.args[0] === FOLDER)).toBe(true)
    // No viewed or outcome line before the user decides.
    expect(ofType('outcome')).toHaveLength(0)
  })

  it('marks an alternative length when the chat result says the reply was cut at max tokens', async () => {
    const store = await loadStore()
    await runContinue(store, { text: GEN, stopReason: 'length' })
    expect(ofType('request')[0].alts[0].finish).toBe('length')
  })

  it('accepting writes viewed and an accepted outcome with the region and what it replaced', async () => {
    const store = await loadStore()
    const p = await runContinue(store)
    await store.useNovel.getState().resolvePending(p.alts[0].text)
    await settle()
    expect(ofType('viewed')).toEqual([expect.objectContaining({ id: p.recId, alts: [0] })])
    const [out] = ofType('outcome')
    // Cursor at the end of 'Mara waited.' (12), a two-character lead, then the 28-character text.
    expect(out).toMatchObject({ id: p.recId, kind: 'accepted', alt: 0, region: { start: 14, end: 14 + GEN.length }, replaced: '' })
    expect(store.useNovel.getState().buffers[P1].text).toBe(`${S1}\n\n${GEN}`)
  })

  it('selecting another alternative adds it to viewed', async () => {
    const store = await loadStore()
    replies.push({ text: GEN, stopReason: 'stop' }, { text: 'She did not answer him.', stopReason: 'stop' })
    await store.useNovel.getState().selectScene('s1')
    await store.useNovel.getState().runAction({ action: 'continue', alternatives: 2 })
    await settle()
    const p = store.useNovel.getState().pending!
    expect(p.alts).toHaveLength(2)
    expect(ofType('request')[0].alts.map((a) => a.index)).toEqual([0, 1])
    store.useNovel.getState().selectAlt(1)
    await store.useNovel.getState().resolvePending(store.useNovel.getState().pending!.alts[1].text)
    await settle()
    expect(ofType('viewed')[0].alts).toEqual([0, 1])
    expect(ofType('outcome')[0]).toMatchObject({ kind: 'accepted', alt: 1 })
  })

  it('a partly kept change is mixed, with the region ending at the final length', async () => {
    const store = await loadStore()
    const p = await runContinue(store)
    const final = `${S1}\n\nHe watched the tide.`
    await store.useNovel.getState().resolvePending(final)
    await settle()
    expect(ofType('outcome')[0]).toMatchObject({ kind: 'mixed', alt: 0, region: { start: 14, end: final.length }, replaced: '' })
    expect(ofType('outcome')[0].id).toBe(p.recId)
  })

  it('rejecting writes viewed and a rejected outcome with no alternative or region', async () => {
    const store = await loadStore()
    const p = await runContinue(store)
    await store.useNovel.getState().resolvePending(p.baseText)
    await settle()
    expect(ofType('viewed')).toHaveLength(1)
    expect(ofType('outcome')).toEqual([expect.objectContaining({ id: p.recId, kind: 'rejected', alt: null, region: null })])
    expect(ofType('final')).toHaveLength(0)
  })

  it('Regenerate abandons the first run and records the second one under its own id', async () => {
    const store = await loadStore()
    const p = await runContinue(store)
    replies.push({ text: 'The gulls wheeled over the water.', stopReason: 'stop' })
    await store.useNovel.getState().regenerate()
    await settle()
    const abandoned = ofType('outcome')
    expect(abandoned).toEqual([expect.objectContaining({ id: p.recId, kind: 'abandoned', alt: null, region: null })])
    const second = store.useNovel.getState().pending!
    expect(second.recId).not.toBe(p.recId)
    expect(ofType('request').map((r) => r.id)).toEqual([p.recId, second.recId])
  })

  it('a stale Accept is abandoned once and writes no accepted outcome', async () => {
    const store = await loadStore()
    const p = await runContinue(store)
    store.useNovel.setState((s) => ({ pending: s.pending ? { ...s.pending, baseHash: 'moved' } : null }))
    await store.useNovel.getState().resolvePending(p.alts[0].text)
    await store.useNovel.getState().resolvePending(p.alts[0].text)
    await settle()
    expect(ofType('outcome')).toEqual([expect.objectContaining({ id: p.recId, kind: 'abandoned' })])
    expect(ofType('viewed')).toHaveLength(0)
    expect(calls.some((c) => c.channel === 'novel:saveFile' && (c.args[0] as { snapshot?: boolean }).snapshot)).toBe(false)
  })

  it('closing the project abandons a change that is still waiting', async () => {
    const store = await loadStore()
    const p = await runContinue(store)
    expect(await store.useNovel.getState().closeProject()).toBe(true)
    await settle()
    expect(ofType('outcome')).toEqual([expect.objectContaining({ id: p.recId, kind: 'abandoned' })])
    // The line went out before main was told to close the project.
    const order = calls.map((c) => c.channel).filter((c) => c === 'novel:record' || c === 'novel:closeProject')
    expect(order.at(-1)).toBe('novel:closeProject')
  })

  describe('final text after an accept', () => {
    const accept = async (store: Awaited<ReturnType<typeof loadStore>>) => {
      const p = await runContinue(store)
      await store.useNovel.getState().resolvePending(p.alts[0].text)
      await settle()
      return p
    }

    it('is written 10 s after the last edit (idle), repeats, and the region follows the text', async () => {
      const store = await loadStore()
      const p = await accept(store)
      expect(ofType('final')).toHaveLength(0)
      const accepted = store.useNovel.getState().buffers[P1].text
      store.useNovel.getState().edit(P1, accepted.replace('He watched', 'He quietly watched'))
      await vi.advanceTimersByTimeAsync(9000)
      expect(ofType('final')).toHaveLength(0)
      // Another edit restarts the 10 s.
      store.useNovel.getState().edit(P1, store.useNovel.getState().buffers[P1].text.replace('tide', 'grey tide'))
      await vi.advanceTimersByTimeAsync(9000)
      expect(ofType('final')).toHaveLength(0)
      await vi.advanceTimersByTimeAsync(1500)
      expect(ofType('final')).toEqual([expect.objectContaining({ id: p.recId, reason: 'idle', text: 'He quietly watched the grey tide come in.' })])
      // A later edit writes another one; the last wins at export.
      store.useNovel.getState().edit(P1, `${store.useNovel.getState().buffers[P1].text} Then.`)
      await vi.advanceTimersByTimeAsync(10500)
      expect(ofType('final')).toHaveLength(2)
    })

    it('is written when the scene is left, and tracking stops', async () => {
      const store = await loadStore()
      const p = await accept(store)
      await store.useNovel.getState().selectScene('s2')
      await settle()
      expect(ofType('final')).toEqual([expect.objectContaining({ id: p.recId, reason: 'left', text: GEN })])
      store.useNovel.getState().edit(P2, 'Oskar slept on.')
      await vi.advanceTimersByTimeAsync(20000)
      expect(ofType('final')).toHaveLength(1)
    })

    it('is written as action when a new action starts on the scene', async () => {
      const store = await loadStore()
      const p = await accept(store)
      replies.push({ text: 'More words follow here.', stopReason: 'stop' })
      await store.useNovel.getState().runAction({ action: 'continue' })
      await settle()
      expect(ofType('final')).toEqual([expect.objectContaining({ id: p.recId, reason: 'action', text: GEN })])
    })

    it('is written as close by the flush on quit, fired without waiting for main', async () => {
      const store = await loadStore()
      const p = await accept(store)
      recordReply = () => new Promise(() => undefined)
      const done = store.flushAll()
      await vi.advanceTimersByTimeAsync(700)
      await expect(done).resolves.toMatchObject({ ok: true })
      expect(ofType('final')).toEqual([expect.objectContaining({ id: p.recId, reason: 'close', text: GEN })])
    })

    it('is text null with reason reloaded after a reload from disk, and tracking stops', async () => {
      const store = await loadStore()
      const p = await accept(store)
      texts[P1] = 'Changed outside.'
      await store.useNovel.getState().reloadFromDisk(P1)
      await settle()
      expect(ofType('final')).toEqual([expect.objectContaining({ id: p.recId, reason: 'reloaded', text: null })])
      store.useNovel.getState().edit(P1, 'Changed outside, and by me.')
      await vi.advanceTimersByTimeAsync(20000)
      expect(ofType('final')).toHaveLength(1)
    })

    it('is text null with reason outside when the region cannot be found any more', async () => {
      const store = await loadStore()
      const p = await accept(store)
      store.useNovel.getState().edit(P1, 'Something else entirely.')
      await vi.advanceTimersByTimeAsync(10500)
      expect(ofType('final')).toEqual([expect.objectContaining({ id: p.recId, reason: 'outside', text: null })])
    })
  })

  describe('cards', () => {
    const summaryReply = JSON.stringify({ summary: 'Mara waits for the ferry.' })

    it('records the raw reply, then accepted when the summary is accepted', async () => {
      const store = await loadStore()
      replies.push({ text: summaryReply, stopReason: 'stop' })
      await store.useNovel.getState().summarizeScene('s1')
      await settle()
      const proposal = store.useNovel.getState().proposal!
      expect(proposal.status).toBe('ready')
      const [req] = ofType('request')
      expect(req).toMatchObject({ id: proposal.recId, action: 'summarizeScene', sceneId: 's1', chapterId: null })
      expect(req.alts).toEqual([expect.objectContaining({ index: 0, generated: summaryReply, finish: 'stop' })])
      await store.useNovel.getState().acceptProposal()
      await settle()
      expect(ofType('outcome')).toEqual([expect.objectContaining({ id: proposal.recId, kind: 'accepted' })])
      expect(ofType('viewed')).toHaveLength(0)
    })

    it('records edited with the edited text, rejected, and abandoned on Retry', async () => {
      const store = await loadStore()
      replies.push({ text: summaryReply, stopReason: 'stop' })
      await store.useNovel.getState().summarizeScene('s1')
      await settle()
      const first = store.useNovel.getState().proposal!
      await store.useNovel.getState().acceptProposal({ ...first, summary: { summary: 'Mara waits, and worries.' } })
      await settle()
      expect(ofType('outcome')).toEqual([expect.objectContaining({ id: first.recId, kind: 'edited', text: 'Mara waits, and worries.' })])

      replies.push({ text: summaryReply, stopReason: 'stop' })
      await store.useNovel.getState().summarizeScene('s2')
      await settle()
      const second = store.useNovel.getState().proposal!
      store.useNovel.getState().rejectProposal()
      await settle()
      expect(ofType('outcome').at(-1)).toMatchObject({ id: second.recId, kind: 'rejected' })

      replies.push({ text: summaryReply, stopReason: 'stop' })
      await store.useNovel.getState().summarizeScene('s2')
      await settle()
      const third = store.useNovel.getState().proposal!
      replies.push({ text: summaryReply, stopReason: 'stop' })
      await store.useNovel.getState().retryProposal()
      await settle()
      expect(ofType('outcome').at(-1)).toMatchObject({ id: third.recId, kind: 'abandoned' })
    })

    it('Check, Update state and Interview write nothing', async () => {
      const store = await loadStore()
      replies.push({ text: JSON.stringify({ findings: [] }), stopReason: 'stop' })
      await store.useNovel.getState().checkScene('s1')
      await settle()
      expect(calls.filter((c) => c.channel === 'chat:start')).toHaveLength(1)
      expect(lines()).toHaveLength(0)
    })
  })

  it('sends nothing when recording is off', async () => {
    st = projectState(false)
    const store = await loadStore()
    const p = await runContinue(store)
    await store.useNovel.getState().resolvePending(p.alts[0].text)
    store.useNovel.getState().edit(P1, `${store.useNovel.getState().buffers[P1].text} More.`)
    await store.useNovel.getState().selectScene('s2')
    await vi.advanceTimersByTimeAsync(20000)
    expect(calls.filter((c) => c.channel === 'novel:record')).toHaveLength(0)
    // The writing itself went on as usual.
    expect(texts[P1]).toContain(GEN)
  })

  it('a failing novel:record never blocks accepting, and tells the user once', async () => {
    recordReply = () => false
    const store = await loadStore()
    const p = await runContinue(store)
    await store.useNovel.getState().resolvePending(p.alts[0].text)
    await store.useNovel.getState().selectScene('s2')
    await settle()
    expect(texts[P1]).toBe(`${S1}\n\n${GEN}`)
    expect(store.useNovel.getState().pending).toBeNull()
    expect(calls.filter((c) => c.channel === 'novel:record').length).toBeGreaterThan(2)
    const recordingToasts = vi.mocked(toast.message).mock.calls.filter((c) => String(c[0]).startsWith('Recording failed'))
    expect(recordingToasts).toHaveLength(1)
  })

  it('a throwing novel:record does not break the run or the accept either', async () => {
    recordReply = () => {
      throw new Error('ipc gone')
    }
    const store = await loadStore()
    const p = await runContinue(store)
    expect(p).toBeTruthy()
    await store.useNovel.getState().resolvePending(p.alts[0].text)
    await settle()
    expect(texts[P1]).toBe(`${S1}\n\n${GEN}`)
    expect(vi.mocked(toast.error)).not.toHaveBeenCalled()
  })

  it('refuses to start a Writing action while something else uses the model', async () => {
    busy.reason = 'A chat reply is running.'
    const store = await loadStore()
    await store.useNovel.getState().selectScene('s1')
    replies.push({ text: GEN })
    await store.useNovel.getState().runAction({ action: 'continue' })
    await store.useNovel.getState().summarizeScene('s1')
    await store.useNovel.getState().checkScene('s1')
    await settle()
    expect(calls.filter((c) => c.channel === 'chat:start')).toHaveLength(0)
    expect(store.useNovel.getState().pending).toBeNull()
    expect(store.useNovel.getState().gen).toBeNull()
    expect(vi.mocked(toast.message)).toHaveBeenCalledWith('A chat reply is running.')
    expect(lines()).toHaveLength(0)
  })
})
