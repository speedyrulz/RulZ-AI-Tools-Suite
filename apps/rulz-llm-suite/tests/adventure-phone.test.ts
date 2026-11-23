// The adventure channels for the window and the phone: the phone's allowlist (no deletes, no save dialog), the
// worldFolder guard in the phone dispatcher, argument checks, the world copied from a Writing project, and the
// desktop-only handlers. Fakes for the novel store and the runs; a real store on a temp folder.
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { PHONE_ADVENTURE_CHANNELS, type Adventure, type AdventureChange, type AdventureSetup } from '../src/shared/adventure'
import { PHONE_EVENTS, PHONE_INVOKE } from '../src/shared/phone'
import type { NovelProjectState } from '../src/shared/novel'
import { adventureDesktopHandlers, adventureHandlers, readWorld, type WorldNovel } from '../src/main/adventure/ipc'
import type { AdventureRuns } from '../src/main/adventure/runs'
import { AdventureStore } from '../src/main/adventure/store'
import { PhoneHandlers, REFUSED } from '../src/main/phone/registry'

const dirs: string[] = []
afterEach(() => {
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true })
})

const BOOK = join(tmpdir(), 'novels', 'Book')

function makeNovel(over: { blocked?: string | null } = {}) {
  const log: string[] = []
  const novel: WorldNovel = {
    listProjects: async () => [{ folder: BOOK, title: 'Book (list)' }],
    openProject: async (folder, owner) => {
      log.push(`open ${folder} ${owner}`)
      return {
        blocked: over.blocked ?? null,
        project: { title: 'Book' },
        outline: { chapters: [{ title: 'One', summary: 'A ferry crosses the bay.' }] },
        bible: [
          { name: 'The Dock', kind: 'place', body: 'A wet dock.' },
          { name: 'Mara', kind: 'character', body: 'A ferrywoman.' }
        ]
      } as unknown as NovelProjectState
    },
    closeProject: async (folder, owner) => void log.push(`close ${folder} ${owner}`)
  }
  return { novel, log }
}

function make(runsOver: Record<string, unknown> = {}) {
  const dir = mkdtempSync(join(tmpdir(), 'adv-phone-'))
  dirs.push(dir)
  const changes: AdventureChange[] = []
  let n = 0
  const store = new AdventureStore({ dir: join(dir, 'a'), newId: () => `id-${String(++n).padStart(8, '0')}`, emit: (c) => changes.push(c) })
  const runs = {
    hasRunning: () => false,
    branchInUse: () => false,
    models: () => [{ id: 'm1', name: 'M', loaded: true }],
    play: async (r: unknown) => ({ played: r }),
    retry: async (r: unknown) => ({ retried: r }),
    list: (id: string) => [{ id }],
    abort: () => undefined,
    dismiss: () => undefined,
    ...runsOver
  } as unknown as AdventureRuns
  const { novel, log } = makeNovel()
  return { store, runs, novel, log, changes }
}

const setup: Partial<AdventureSetup> = { title: 'Lantern', genre: 'Fantasy', premise: 'A keeper.', character: 'Wren', tone: 'Dry', length: 'medium', choiceCount: 3, modelId: null, temperature: null }

/** The phone dispatcher with the adventure handlers registered the way main/index.ts does. */
function phoneWith(t: ReturnType<typeof make>, known: string[] = [BOOK]) {
  const table = adventureHandlers({ store: t.store, runs: t.runs, novel: t.novel })
  const handlers = new PhoneHandlers({ projectFolders: async () => known, ownsProject: () => false })
  for (const channel of PHONE_ADVENTURE_CHANNELS) handlers.register(channel, ((owner: string, ...args: unknown[]) => table[channel](owner, ...args)) as never)
  return { handlers, table }
}

describe('the phone allowlist', () => {
  it('has every adventure channel except the desktop-only ones, and the two events', () => {
    for (const c of PHONE_ADVENTURE_CHANNELS) expect(PHONE_INVOKE.has(c)).toBe(true)
    for (const c of ['adventure:delete', 'adventure:deleteBranch', 'adventure:saveExport']) expect(PHONE_INVOKE.has(c)).toBe(false)
    expect(PHONE_EVENTS.has('adventure:job')).toBe(true)
    expect(PHONE_EVENTS.has('adventure:changed')).toBe(true)
    expect(PHONE_ADVENTURE_CHANNELS).toHaveLength(14)
  })

  it('refuses delete, deleteBranch and saveExport from a phone, even with a handler present', async () => {
    const t = make()
    const { handlers } = phoneWith(t)
    const adv = await t.store.create(setup)
    for (const [channel, args] of [
      ['adventure:delete', [adv.id]],
      ['adventure:deleteBranch', [adv.id, 'n', 1]],
      ['adventure:saveExport', [adv.id]]
    ] as const) {
      await expect(handlers.invoke('phone:1', channel, [...args])).rejects.toThrow(REFUSED)
      expect(() => handlers.register(channel as never, (() => undefined) as never)).toThrow('is not a phone channel')
    }
    expect((await t.store.list()).map((m) => m.id)).toEqual([adv.id])
  })

  it('answers every allowed channel through the dispatcher', async () => {
    const t = make()
    const { handlers } = phoneWith(t)
    const created = (await handlers.invoke('phone:1', 'adventure:create', [setup])) as Adventure
    expect(created.setup.title).toBe('Lantern')
    expect(((await handlers.invoke('phone:1', 'adventure:list', [])) as unknown[]).length).toBe(1)
    expect(((await handlers.invoke('phone:1', 'adventure:get', [created.id])) as Adventure).id).toBe(created.id)
    expect(await handlers.invoke('phone:1', 'adventure:starters', [])).toHaveLength(8)
    expect(await handlers.invoke('phone:1', 'adventure:models', [])).toEqual([{ id: 'm1', name: 'M', loaded: true }])
    expect(await handlers.invoke('phone:1', 'adventure:jobs', [created.id])).toEqual([{ id: created.id }])
    expect(await handlers.invoke('phone:1', 'adventure:play', [{ adventureId: created.id, parentId: null, input: { kind: 'start' } }])).toMatchObject({ played: { parentId: null } })
    expect(await handlers.invoke('phone:1', 'adventure:retry', [{ adventureId: created.id, nodeId: 'n', what: 'choices' }])).toMatchObject({ retried: { what: 'choices' } })
    expect(await handlers.invoke('phone:1', 'adventure:abort', ['job'])).toBeUndefined()
    expect(await handlers.invoke('phone:1', 'adventure:dismiss', ['job'])).toBeUndefined()
    const ch = (await handlers.invoke('phone:1', 'adventure:updateSetup', [created.id, { title: 'Renamed' }])) as AdventureChange
    expect(ch.setup?.title).toBe('Renamed')
    expect(await handlers.invoke('phone:1', 'adventure:exportMarkdown', [created.id])).toContain('# Renamed')
  })
})

describe('adventure:create and the worldFolder guard', () => {
  it('lets a phone start from a Writing project it never opened; the digest goes into setup.world', async () => {
    const t = make()
    const { handlers } = phoneWith(t)
    const adv = (await handlers.invoke('phone:1', 'adventure:create', [{ ...setup, world: { folder: 'x', title: 'evil', text: 'evil' } }, BOOK])) as Adventure
    expect(adv.setup.world?.title).toBe('Book')
    expect(adv.setup.world?.folder).toBe(BOOK)
    expect(adv.setup.world?.text).toContain('A ferry crosses the bay.')
    expect(adv.setup.world?.text).not.toContain('evil')
    // Characters come before places.
    const text = adv.setup.world?.text as string
    expect(text.indexOf('Mara')).toBeLessThan(text.indexOf('The Dock'))
    // Opened under its own owner for a moment, and closed again.
    expect(t.log).toEqual([`open ${BOOK} adventure:${adv.id}`, `close ${BOOK} adventure:${adv.id}`])
    expect((await t.store.get(adv.id)).setup.world?.text).toBe(text)
  })

  it('refuses a folder that is not absolute or not one of the listed projects, before anything is opened', async () => {
    const t = make()
    const { handlers } = phoneWith(t)
    await expect(handlers.invoke('phone:1', 'adventure:create', [setup, 'Book'])).rejects.toThrow('must be a full path')
    await expect(handlers.invoke('phone:1', 'adventure:create', [setup, 42])).rejects.toThrow('must be a full path')
    await expect(handlers.invoke('phone:1', 'adventure:create', [setup, join(tmpdir(), 'elsewhere')])).rejects.toThrow('only use projects in the projects folder')
    await expect(handlers.invoke('phone:1', 'adventure:create', [setup, join(BOOK, '..', '..', 'secrets')])).rejects.toThrow('only use projects in the projects folder')
    expect(t.log).toEqual([])
    expect(await t.store.list()).toEqual([])
    // No folder (undefined or null) is the "own idea" path.
    await handlers.invoke('phone:1', 'adventure:create', [setup])
    await handlers.invoke('phone:1', 'adventure:create', [setup, null])
    expect((await t.store.list()).length).toBe(2)
  })

  it('accepts the same folder written in another case or with ..  on Windows, like novel:openProject', async () => {
    const t = make()
    const { handlers } = phoneWith(t)
    const variant = join(BOOK, '..', 'Book')
    const adv = (await handlers.invoke('phone:1', 'adventure:create', [setup, variant])) as Adventure
    expect(adv.setup.world?.folder).toBe(BOOK)
  })

  it('the handler checks the list again for the window, and refuses a blocked project; the project is closed anyway', async () => {
    const t = make()
    const table = adventureHandlers({ store: t.store, runs: t.runs, novel: t.novel })
    await expect(table['adventure:create']('window', setup, join(tmpdir(), 'elsewhere'))).rejects.toThrow('not found in the projects folder')
    const blocked = makeNovel({ blocked: 'project.json is damaged' })
    await expect(readWorld(blocked.novel, BOOK, 'adventure:x')).rejects.toThrow('cannot be read: project.json is damaged')
    expect(blocked.log).toEqual([`open ${BOOK} adventure:x`, `close ${BOOK} adventure:x`])
    expect(await t.store.list()).toEqual([])
  })
})

describe('argument checks', () => {
  it('refuses ids that are not plain ids before a path is built, and bad shapes', async () => {
    const t = make()
    const table = adventureHandlers({ store: t.store, runs: t.runs, novel: t.novel })
    await expect(table['adventure:get']('window', '../../x')).rejects.toThrow('not an adventure id')
    await expect(table['adventure:exportMarkdown']('window', 'a\\b')).rejects.toThrow('not an adventure id')
    await expect(table['adventure:updateSetup']('window', 'short', {})).rejects.toThrow('not an adventure id')
    await expect(table['adventure:updateSetup']('window', 'long-enough-id', null)).rejects.toThrow('settings are missing')
    await expect(table['adventure:setCurrent']('window', 'long-enough-id', '')).rejects.toThrow('passage is missing')
    await expect(table['adventure:edit']('window', null)).rejects.toThrow('edit is missing')
    await expect(table['adventure:edit']('window', { adventureId: 'long-enough-id', nodeId: 'n' })).rejects.toThrow('which version')
    await expect(table['adventure:edit']('window', { adventureId: '../x', nodeId: 'n', rev: 1 })).rejects.toThrow('not an adventure id')
    await expect(table['adventure:create']('window', null)).rejects.toThrow('setup is missing')
    await expect(table['adventure:abort']('window', '')).rejects.toThrow('job id is missing')
    await expect(table['adventure:jobs']('window', 'x/y')).rejects.toThrow('not an adventure id')
  })

  it('exports the path to the given node, or the current one, and refuses an unknown node', async () => {
    const t = make()
    const table = adventureHandlers({ store: t.store, runs: t.runs, novel: t.novel })
    const adv = await t.store.create(setup)
    const a = (await t.store.addNode(adv.id, { parentId: null, input: { kind: 'start' }, text: 'Opening.' })).node
    const b = (await t.store.addNode(adv.id, { parentId: a.id, input: { kind: 'action', text: 'Wait' }, text: 'Later.' })).node
    expect(await table['adventure:exportMarkdown']('window', adv.id)).toContain('Later.')
    expect(await table['adventure:exportMarkdown']('window', adv.id, a.id)).not.toContain('Later.')
    expect(await table['adventure:exportMarkdown']('window', adv.id, b.id)).toContain('*You: Wait*')
    await expect(table['adventure:exportMarkdown']('window', adv.id, 'nope')).rejects.toThrow('not found')
  })
})

describe('the desktop-only handlers', () => {
  it('delete is refused while a job runs on the adventure, then goes to the Recycle Bin path', async () => {
    let running = true
    const t = make({ hasRunning: () => running })
    const trashed: string[] = []
    const dir = mkdtempSync(join(tmpdir(), 'adv-del-'))
    dirs.push(dir)
    const store = new AdventureStore({ dir: join(dir, 'a'), trash: async (p) => void trashed.push(p) })
    const desk = adventureDesktopHandlers({ store, runs: t.runs, novel: t.novel, chooseSavePath: async () => null, writeText: async () => undefined })
    const adv = await store.create(setup)
    await expect(desk['adventure:delete']('window', adv.id)).rejects.toThrow('being written')
    running = false
    await desk['adventure:delete']('window', adv.id)
    expect(trashed).toHaveLength(1)
    expect(trashed[0]).toContain(adv.id)
    await expect(desk['adventure:delete']('window', '../x')).rejects.toThrow('not an adventure id')
  })

  it('deleteBranch asks the runs whether a job uses the branch', async () => {
    const asked: string[][] = []
    const t = make({ branchInUse: (_id: string, ids: Set<string>) => (asked.push([...ids]), true) })
    const desk = adventureDesktopHandlers({ store: t.store, runs: t.runs, novel: t.novel, chooseSavePath: async () => null, writeText: async () => undefined })
    const adv = await t.store.create(setup)
    const root = (await t.store.addNode(adv.id, { parentId: null, input: { kind: 'start' }, text: 'O.' })).node
    const kid = (await t.store.addNode(adv.id, { parentId: root.id, input: { kind: 'action', text: 'x' }, text: 'K.' })).node
    await expect(desk['adventure:deleteBranch']('window', adv.id, kid.id, kid.rev)).rejects.toThrow('being written')
    expect(asked).toEqual([[kid.id]])
    await expect(desk['adventure:deleteBranch']('window', adv.id, kid.id, 'x')).rejects.toThrow('which version')
    const free = make()
    const desk2 = adventureDesktopHandlers({ store: free.store, runs: free.runs, novel: free.novel, chooseSavePath: async () => null, writeText: async () => undefined })
    const adv2 = await free.store.create(setup)
    const r2 = (await free.store.addNode(adv2.id, { parentId: null, input: { kind: 'start' }, text: 'O.' })).node
    const k2 = (await free.store.addNode(adv2.id, { parentId: r2.id, input: { kind: 'action', text: 'x' }, text: 'K.' })).node
    const ch = (await desk2['adventure:deleteBranch']('window', adv2.id, k2.id, k2.rev)) as AdventureChange
    expect(ch.removedIds).toEqual([k2.id])
    expect(ch.currentId).toBe(r2.id)
  })

  it('saveExport asks for a path with a Windows-safe name, writes the Markdown, and returns null on cancel', async () => {
    const t = make()
    const asked: string[] = []
    const written: { path: string; text: string }[] = []
    let answer: string | null = 'C:\\out\\story.md'
    const desk = adventureDesktopHandlers({
      store: t.store,
      runs: t.runs,
      novel: t.novel,
      chooseSavePath: async (name) => (asked.push(name), answer),
      writeText: async (path, text) => void written.push({ path, text })
    })
    const adv = await t.store.create({ ...setup, title: 'What? A "Story": a/b' })
    await t.store.addNode(adv.id, { parentId: null, input: { kind: 'start' }, text: 'Opening.' })
    expect(await desk['adventure:saveExport']('window', adv.id)).toBe('C:\\out\\story.md')
    expect(asked).toEqual(['What A Story a b.md'])
    expect(written).toHaveLength(1)
    expect(written[0].path).toBe('C:\\out\\story.md')
    expect(written[0].text).toContain('# What? A "Story": a/b')
    expect(written[0].text).toContain('Opening.')
    answer = null
    expect(await desk['adventure:saveExport']('window', adv.id)).toBeNull()
    expect(written).toHaveLength(1)
  })
})
