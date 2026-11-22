import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { NovelTextFile } from '../src/shared/novel'

// The editor store with a pretend computer: each readFile / saveFile answer is a promise the test settles.

const prefs: Record<string, unknown> = {}
vi.mock('../src/lib/native', () => ({
  loadPref: async (key: string, fallback: unknown) => (key in prefs ? structuredClone(prefs[key]) : fallback),
  savePref: async (key: string, value: unknown) => void (prefs[key] = structuredClone(value))
}))
vi.mock('../src/stores/link', () => ({ currentComputer: () => ({ id: 'pc1' }) }))
vi.mock('sonner', () => ({ toast: { error: vi.fn(), success: vi.fn() } }))

type Call = { channel: string; args: unknown[]; resolve: (v: unknown) => void; reject: (e: Error) => void }
const calls: Call[] = []
vi.mock('../src/lib/remote', () => ({
  link: {
    invoke: (channel: string, ...args: unknown[]) => new Promise((resolve, reject) => calls.push({ channel, args, resolve, reject })),
    on: () => () => undefined
  }
}))

const { createEditor, resetDraftCache } = await import('../src/stores/editor')
const { draftKey } = await import('../src/lib/autosave')

const FOLDER = 'C:\\Novels\\Ferry'
const PATH = 'manuscript/a.md'
const file = (text: string, hash: string): NovelTextFile => ({ path: PATH, text, hash, meta: { bom: false, eol: 'lf' }, readOnly: false, journal: null })
const flush = (): Promise<void> => new Promise((r) => setTimeout(r, 0))
const take = (channel: string): Call => {
  const i = calls.findIndex((c) => c.channel === channel)
  if (i < 0) throw new Error(`no pending ${channel}`)
  return calls.splice(i, 1)[0]
}

async function opened(text = 'Disk text.', hash = 'h1') {
  const store = createEditor()
  const p = store.getState().open(FOLDER, PATH)
  await flush()
  take('novel:readFile').resolve(file(text, hash))
  await p
  return store
}

beforeEach(() => {
  calls.length = 0
  resetDraftCache()
  for (const k of Object.keys(prefs)) delete prefs[k]
  vi.useRealTimers()
})

describe('the draft offer (1)', () => {
  it('keeps typing out and the offered draft intact until the owner chooses', async () => {
    prefs.drafts = { [draftKey(FOLDER, PATH, 'pc1')]: { text: 'Old draft.', baseHash: 'h1', savedAt: 1 } }
    const store = await opened('Disk text.', 'h1')
    expect(store.getState().offer?.draft.text).toBe('Old draft.')
    store.getState().edit('typed anyway')
    expect(store.getState().buffer?.text).toBe('Disk text.')
    expect(await store.getState().applyText('accepted')).toBe(false)
    expect(calls.filter((c) => c.channel === 'novel:saveFile')).toHaveLength(0)
    store.getState().restoreDraft()
    expect(store.getState().buffer?.text).toBe('Old draft.')
    store.getState().edit('Old draft. And more.')
    expect(store.getState().buffer?.text).toBe('Old draft. And more.')
  })

  it('finds a draft saved under the old key (no computer id)', async () => {
    prefs.drafts = { [draftKey(FOLDER, PATH)]: { text: 'Legacy.', baseHash: 'h1', savedAt: 1 } }
    const store = await opened()
    expect(store.getState().offer?.draft.text).toBe('Legacy.')
  })

  it('does not offer another computer’s draft for the same path', async () => {
    prefs.drafts = { [draftKey(FOLDER, PATH, 'pc2')]: { text: 'Other computer.', baseHash: 'h1', savedAt: 1 } }
    const store = await opened()
    expect(store.getState().offer).toBeNull()
  })
})

describe('loading the computer’s text (2, 5)', () => {
  it('keeps text typed while the read was pending, and shows the banner', async () => {
    const store = await opened('Disk text.', 'h1')
    store.getState().edit('Disk text. Typed.')
    const p = store.getState().loadFromDisk()
    await flush()
    store.getState().edit('Disk text. Typed more.')
    take('novel:readFile').resolve(file('Changed on computer.', 'h2'))
    await p
    expect(store.getState().buffer?.text).toBe('Disk text. Typed more.')
    expect(store.getState().buffer?.changedOnDisk).toBe('changed')
  })

  it('replaces the text when nothing was typed during the read', async () => {
    const store = await opened('Disk text.', 'h1')
    store.getState().edit('Disk text. Typed.')
    const p = store.getState().loadFromDisk()
    await flush()
    take('novel:readFile').resolve(file('Changed on computer.', 'h2'))
    await p
    expect(store.getState().buffer).toMatchObject({ text: 'Changed on computer.', baseHash: 'h2' })
  })

  it('drops a change-triggered reload that belongs to a file that was left', async () => {
    const store = await opened('Disk text.', 'h1')
    store.getState().onChange({ kind: 'text', folder: FOLDER, path: PATH, hash: 'h2' })
    await flush()
    const read = take('novel:readFile')
    store.getState().drop()
    read.resolve(file('Changed on computer.', 'h2'))
    await flush()
    expect(store.getState().buffer).toBeNull()
  })

  it('only the newest read applies', async () => {
    const store = await opened('Disk text.', 'h1')
    store.getState().onChange({ kind: 'text', folder: FOLDER, path: PATH, hash: 'h2' })
    await flush()
    const first = take('novel:readFile')
    store.getState().onChange({ kind: 'text', folder: FOLDER, path: PATH, hash: 'h3' })
    await flush()
    const second = take('novel:readFile')
    second.resolve(file('Third.', 'h3'))
    await flush()
    first.resolve(file('Second.', 'h2'))
    await flush()
    expect(store.getState().buffer?.text).toBe('Third.')
  })

  it('reloads a clean buffer when a change arrives, and keeps an edited one', async () => {
    const store = await opened('Disk text.', 'h1')
    store.getState().onChange({ kind: 'text', folder: FOLDER, path: PATH, hash: 'h2' })
    await flush()
    take('novel:readFile').resolve(file('Changed on computer.', 'h2'))
    await flush()
    expect(store.getState().buffer?.text).toBe('Changed on computer.')
    store.getState().edit('Mine.')
    store.getState().onChange({ kind: 'text', folder: FOLDER, path: PATH, hash: 'h3' })
    expect(store.getState().buffer).toMatchObject({ text: 'Mine.', changedOnDisk: 'changed' })
    expect(calls.filter((c) => c.channel === 'novel:readFile')).toHaveLength(0)
  })
})

describe('changes during our own save (6)', () => {
  it('reloads after the save when the computer changed the file after our write', async () => {
    const store = await opened('Disk text.', 'h1')
    store.getState().edit('Mine.')
    const saving = store.getState().saveNow()
    await flush()
    const save = take('novel:saveFile')
    store.getState().onChange({ kind: 'text', folder: FOLDER, path: PATH, hash: 'h2' })
    store.getState().onChange({ kind: 'text', folder: FOLDER, path: PATH, hash: 'h3' })
    save.resolve({ ok: true, hash: 'h2' })
    await saving
    await flush()
    take('novel:readFile').resolve(file('Their later text.', 'h3'))
    await flush()
    expect(store.getState().buffer).toMatchObject({ text: 'Their later text.', baseHash: 'h3' })
  })
})

describe('leaving a file (3)', () => {
  it('blocks typing during the save and keeps a draft of anything unsaved', async () => {
    const store = await opened('Disk text.', 'h1')
    store.getState().edit('Mine.')
    const leaving = store.getState().leave()
    await flush()
    const save = take('novel:saveFile')
    store.getState().edit('Typed while leaving.')
    expect(store.getState().buffer?.text).toBe('Mine.')
    save.reject(new Error('Lost the connection.'))
    await leaving
    expect(store.getState().buffer).toBeNull()
    const saved = (prefs.drafts as Record<string, { text: string }>)[draftKey(FOLDER, PATH, 'pc1')]
    expect(saved?.text).toBe('Mine.')
  })
})

const PATH2 = 'manuscript/b.md'

describe('opening files in quick succession (open race)', () => {
  it('ends on the file asked for last, even when an older open was waiting for leave()', async () => {
    const store = await opened('Scene one.', 'h1')
    store.getState().edit('Scene one, edited.')
    // Open scene 2: it waits for the save of scene 1.
    const toTwo = store.getState().open(FOLDER, PATH2)
    await flush()
    const save = take('novel:saveFile')
    // Meanwhile the screen asks for scene 1 again.
    const backToOne = store.getState().open(FOLDER, PATH)
    save.resolve({ ok: true, hash: 'h1b' })
    await flush()
    await flush()
    // Only one file is read: the one asked for last.
    const reads = calls.filter((c) => c.channel === 'novel:readFile')
    expect(reads).toHaveLength(1)
    expect(reads[0].args[1]).toBe(PATH)
    take('novel:readFile').resolve(file('Scene one, edited.', 'h1b'))
    expect(await toTwo).toBe(false)
    expect(await backToOne).toBe(true)
    expect(store.getState().buffer?.path).toBe(PATH)
  })

  it('does nothing when the file asked for is already the one asked for last', async () => {
    const store = await opened()
    expect(await store.getState().open(FOLDER, PATH)).toBe(true)
    expect(calls).toHaveLength(0)
  })
})

describe('conflicts keep the file open', () => {
  async function conflicted() {
    const store = await opened('Disk text.', 'h1')
    store.getState().edit('Mine.')
    const saving = store.getState().saveNow()
    await flush()
    take('novel:saveFile').resolve({ ok: false, reason: 'conflict', diskHash: 'h9', diskText: 'Theirs.' })
    await saving
    return store
  }

  it('refuses to leave or open another file, so the conflict is answered first', async () => {
    const store = await conflicted()
    expect(store.getState().buffer?.conflict).not.toBeNull()
    expect(await store.getState().leave()).toBe(false)
    expect(store.getState().buffer?.path).toBe(PATH)
    expect(await store.getState().open(FOLDER, PATH2)).toBe(false)
    expect(store.getState().buffer?.path).toBe(PATH)
    expect(calls.filter((c) => c.channel === 'novel:readFile')).toHaveLength(0)
  })

  it('keep mine saves with a snapshot request on the new base', async () => {
    const store = await conflicted()
    const kept = store.getState().keepMine()
    await flush()
    const save = take('novel:saveFile')
    expect(save.args[0]).toMatchObject({ text: 'Mine.', baseHash: 'h9', snapshot: true })
    expect(save.args[0]).not.toHaveProperty('force')
    save.resolve({ ok: true, hash: 'h10' })
    await kept
    expect(store.getState().buffer).toMatchObject({ baseHash: 'h10', savedText: 'Mine.' })
  })
})

describe('accepting an AI result (applyText)', () => {
  it('saves with a snapshot request and keeps the text when it saves', async () => {
    const store = await opened('Disk text.', 'h1')
    const done = store.getState().applyText('AI text.')
    await flush()
    const save = take('novel:saveFile')
    expect(save.args[0]).toMatchObject({ text: 'AI text.', snapshot: true })
    save.resolve({ ok: true, hash: 'h2' })
    expect(await done).toBe(true)
    expect(store.getState().buffer?.text).toBe('AI text.')
  })

  it('puts the previous text back when the save fails, and does not autosave the AI text later', async () => {
    const store = await opened('Disk text.', 'h1')
    const done = store.getState().applyText('AI text.')
    await flush()
    take('novel:saveFile').reject(new Error('Lost the connection.'))
    expect(await done).toBe(false)
    expect(store.getState().buffer?.text).toBe('Disk text.')
    expect(store.getState().buffer?.snapshotNext).toBe(false)
  })

  it('puts the previous text back when the save conflicts', async () => {
    const store = await opened('Disk text.', 'h1')
    const done = store.getState().applyText('AI text.')
    await flush()
    take('novel:saveFile').resolve({ ok: false, reason: 'conflict', diskHash: 'h9', diskText: 'Theirs.' })
    expect(await done).toBe(false)
    expect(store.getState().buffer?.text).toBe('Disk text.')
    expect(store.getState().buffer?.conflict).not.toBeNull()
  })
})

describe('a file that was deleted', () => {
  it('lets go without saving, and forgets its draft', async () => {
    const { discardDraftFor } = await import('../src/stores/editor')
    const store = await opened('Disk text.', 'h1')
    store.getState().edit('Typed.')
    await flush()
    store.getState().drop()
    await discardDraftFor(FOLDER, PATH)
    expect(store.getState().buffer).toBeNull()
    expect(calls.filter((c) => c.channel === 'novel:saveFile')).toHaveLength(0)
    // Opening the file again offers no draft.
    const again = await opened('Disk text.', 'h1')
    expect(again.getState().offer).toBeNull()
  })
})
