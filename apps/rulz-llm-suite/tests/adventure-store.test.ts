// AdventureStore on a temp folder: ids, serialised writes, corrupt and newer files, revs and conflicts, precise
// changes, state carried down, deleteBranch and delete. No model, no Electron.
import { mkdtempSync, readdirSync, rmSync, writeFileSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { ADVENTURE_FORMAT, applyChange, metaOf, noChanges, type Adventure, type AdventureChange, type AdventureSetup } from '../src/shared/adventure'
import { AdventureStore, CONFLICT_MESSAGE, NEWER_FORMAT_MESSAGE, checkAdventureId, normalizeSetup } from '../src/main/adventure/store'

const dirs: string[] = []
afterEach(() => {
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true })
})

const setup = (over: Partial<AdventureSetup> = {}): Partial<AdventureSetup> => ({
  title: 'Lantern',
  genre: 'Fantasy',
  premise: 'A keeper.',
  character: 'Wren',
  tone: 'Dry',
  length: 'medium',
  choiceCount: 3,
  modelId: null,
  temperature: null,
  ...over
})

function make() {
  const dir = mkdtempSync(join(tmpdir(), 'adv-store-'))
  dirs.push(dir)
  let t = 1000
  let n = 0
  const changes: AdventureChange[] = []
  const trashed: string[] = []
  const store = new AdventureStore({
    dir: join(dir, 'adventures'),
    now: () => ++t,
    newId: () => `id-${String(++n).padStart(8, '0')}`,
    trash: async (p) => {
      trashed.push(p)
      rmSync(p, { force: true })
    },
    emit: (c) => changes.push(c)
  })
  return { store, dir: join(dir, 'adventures'), changes, trashed }
}

/** An adventure with a root and two children: root -> a (choice) and root -> b (action); a -> grand. */
async function tree() {
  const m = make()
  const adv = await m.store.create(setup())
  const root = (await m.store.addNode(adv.id, { parentId: null, input: { kind: 'start' }, text: 'Opening.' })).node
  await m.store.setChoosing(adv.id, root.id, {
    choices: ['Knock', 'Climb'],
    ended: false,
    changes: { ...noChanges(), addItems: ['lamp'], setStats: [{ name: 'Health', value: 10 }], addFlags: ['at the gate'] }
  })
  const rootNow = (await m.store.get(adv.id)).nodes[root.id]
  const choiceId = rootNow.choices[0].id
  const a = (await m.store.addNode(adv.id, { parentId: root.id, input: { kind: 'choice', choiceId, text: 'Knock' }, text: 'A voice answers.' })).node
  await m.store.setChoosing(adv.id, a.id, { choices: ['Enter', 'Leave'], ended: false, changes: { ...noChanges(), addItems: ['key'], setStats: [{ name: 'health', value: 9 }] } })
  const b = (await m.store.addNode(adv.id, { parentId: root.id, input: { kind: 'action', text: 'Climb the wall' }, text: 'You climb.' })).node
  const aNow = (await m.store.get(adv.id)).nodes[a.id]
  const grand = (await m.store.addNode(adv.id, { parentId: a.id, input: { kind: 'choice', choiceId: aNow.choices[0].id, text: 'Enter' }, text: 'Inside.' })).node
  await m.store.setChoosing(adv.id, grand.id, { choices: ['Sit'], ended: false, changes: { ...noChanges(), removeItems: ['lamp'] } })
  m.changes.length = 0
  return { ...m, id: adv.id, rootId: root.id, aId: a.id, bId: b.id, grandId: grand.id }
}

describe('ids and setup', () => {
  it('refuses anything that is not a plain id before a path is built', async () => {
    const { store } = make()
    for (const bad of ['', '../x', 'a/b', 'a\\b', 'short', 'x'.repeat(65), 'has space1234', 'dots..1234', null, 42]) {
      expect(() => checkAdventureId(bad as never)).toThrow('not an adventure id')
    }
    await expect(store.get('../../etc/passwd')).rejects.toThrow('not an adventure id')
    expect(() => store.delete('..\\..\\x')).toThrow('not an adventure id')
    expect(checkAdventureId('0f8fad5b-d9cb-469f-a165-70867728950e')).toBe('0f8fad5b-d9cb-469f-a165-70867728950e')
  })

  it('creates with random-looking ids, a normalised setup, and ignores a world sent in the setup', async () => {
    const { store, dir, changes } = make()
    const adv = await store.create({ ...setup({ title: '  ', choiceCount: 9, temperature: 7, length: 'huge' as never }), world: { folder: 'x', title: 'evil', text: 'evil' } } as never)
    expect(adv.id).toBe('id-00000001')
    expect(adv.setup.title).toBe('Untitled adventure')
    expect(adv.setup.choiceCount).toBe(4)
    expect(adv.setup.temperature).toBe(2)
    expect(adv.setup.length).toBe('medium')
    expect(adv.setup.world).toBeUndefined()
    expect(adv.format).toBe(ADVENTURE_FORMAT)
    expect(readdirSync(dir)).toEqual(['id-00000001.json'])
    expect(changes).toHaveLength(1)
    expect(changes[0]).toMatchObject({ id: adv.id, rev: 1, setup: adv.setup, meta: { id: adv.id, nodeCount: 0 } })
    const withWorld = await store.create(setup(), { folder: 'C:\\b', title: 'Book', text: 'T' })
    expect(withWorld.setup.world).toEqual({ folder: 'C:\\b', title: 'Book', text: 'T' })
    expect(normalizeSetup({ temperature: null, modelId: 'm1' }).modelId).toBe('m1')
  })
})

describe('listing, corrupt and newer files', () => {
  it('lists adventures newest first, leaves a corrupt file out and keeps it aside', async () => {
    const { store, dir } = make()
    const one = await store.create(setup({ title: 'One' }))
    const two = await store.create(setup({ title: 'Two' }))
    writeFileSync(join(dir, 'broken-file-1234.json'), '{ not json')
    writeFileSync(join(dir, 'notes.txt'), 'x')
    writeFileSync(join(dir, 'shape-wrong-1234.json'), JSON.stringify({ hello: 1 }))
    const list = await store.list()
    expect(list.map((m) => m.title)).toEqual(['Two', 'One'])
    expect(list.map((m) => m.id)).toEqual([two.id, one.id])
    const names = readdirSync(dir)
    expect(names.some((n) => n.startsWith('broken-file-1234.json.corrupt-'))).toBe(true)
    expect(names.some((n) => n.startsWith('shape-wrong-1234.json.corrupt-'))).toBe(true)
    expect(names).not.toContain('broken-file-1234.json')
    await expect(store.get('broken-file-1234')).rejects.toThrow('not found')
  })

  it('lists an adventure from a newer format but refuses to open or change it', async () => {
    const { store, dir } = make()
    const adv = await store.create(setup({ title: 'Future' }))
    const file = join(dir, `${adv.id}.json`)
    writeFileSync(file, JSON.stringify({ ...adv, format: ADVENTURE_FORMAT + 1 }))
    expect((await store.list()).map((m) => m.title)).toEqual(['Future'])
    await expect(store.get(adv.id)).rejects.toThrow(NEWER_FORMAT_MESSAGE)
    await expect(store.updateSetup(adv.id, { title: 'x' })).rejects.toThrow(NEWER_FORMAT_MESSAGE)
    await expect(store.addNode(adv.id, { parentId: null, input: { kind: 'start' }, text: 'x' })).rejects.toThrow(NEWER_FORMAT_MESSAGE)
  })
})

describe('saving', () => {
  it('raises the rev on every save and emits a precise change, returning the same object', async () => {
    const { store, changes } = make()
    const adv = await store.create(setup())
    changes.length = 0
    const added = await store.addNode(adv.id, { parentId: null, input: { kind: 'start' }, text: 'Opening.' })
    expect(changes).toHaveLength(1)
    expect(changes[0]).toBe(added.change)
    expect(added.change).toMatchObject({ id: adv.id, rev: 2, rootId: added.node.id, currentId: added.node.id })
    expect(added.change.nodes).toEqual([added.node])
    expect(added.change.meta).toMatchObject({ nodeCount: 1, depth: 1, ended: false })
    expect(added.node).toMatchObject({ parentId: null, choices: [], ended: false, rev: 1, edited: false, changes: noChanges() })

    const upd = await store.updateSetup(adv.id, { title: 'New title', tone: 'Grim', premise: 'IGNORED', character: 'IGNORED', world: 'x', length: 'bogus', choiceCount: 'x' } as never)
    expect(changes[1]).toBe(upd)
    expect(upd.rev).toBe(3)
    expect(upd.setup).toMatchObject({ title: 'New title', tone: 'Grim', premise: 'A keeper.', character: 'Wren', length: 'medium', choiceCount: 3 })
    expect(upd.nodes).toBeUndefined()
    expect(upd.meta?.title).toBe('New title')
    expect((await store.updateSetup(adv.id, { title: '   ' })).setup?.title).toBe('New title')

    const cur = await store.setCurrent(adv.id, added.node.id)
    expect(cur.currentId).toBe(added.node.id)
    expect(cur.nodes).toBeUndefined()
    expect((await store.get(adv.id)).nodes[added.node.id].rev).toBe(1)
    await expect(store.setCurrent(adv.id, 'nope')).rejects.toThrow('not found')
    await expect(store.setCurrent(adv.id, '__proto__')).rejects.toThrow('not found')
    expect((await store.get(adv.id)).rev).toBe(5)
  })

  it('lets a client patch what it holds with applyChange', async () => {
    const { store, changes } = make()
    const adv = await store.create(setup())
    let held: Adventure = await store.get(adv.id)
    const root = await store.addNode(adv.id, { parentId: null, input: { kind: 'start' }, text: 'Opening.' })
    await store.setChoosing(adv.id, root.node.id, { choices: ['A', 'B'], ended: false, changes: noChanges() })
    await store.updateSetup(adv.id, { title: 'Patched' })
    for (const c of changes.slice(1)) held = applyChange(held, c)
    const disk = await store.get(adv.id)
    expect(held.nodes).toEqual(disk.nodes)
    expect(held.setup).toEqual(disk.setup)
    expect(held.rev).toBe(disk.rev)
    expect(held.rootId).toBe(disk.rootId)
    expect(held.currentId).toBe(disk.currentId)
    expect(metaOf(disk).title).toBe('Patched')
  })

  it('serialises writes to one adventure: concurrent saves all land with distinct revs', async () => {
    const { store } = make()
    const adv = await store.create(setup())
    const results = await Promise.all(Array.from({ length: 12 }, (_, i) => store.updateSetup(adv.id, { tone: `t${i}` })))
    expect(results.map((r) => r.rev)).toEqual(Array.from({ length: 12 }, (_, i) => i + 2))
    expect((await store.get(adv.id)).rev).toBe(13)
  })

  it('refuses a second opening, a parent that is gone, and mismatched inputs', async () => {
    const { store } = make()
    const adv = await store.create(setup())
    await expect(store.addNode(adv.id, { parentId: null, input: { kind: 'action', text: 'x' }, text: 'x' })).rejects.toThrow('Only the opening')
    const root = await store.addNode(adv.id, { parentId: null, input: { kind: 'start' }, text: 'Opening.' })
    await expect(store.addNode(adv.id, { parentId: null, input: { kind: 'start' }, text: 'again' })).rejects.toThrow('already been written')
    await expect(store.addNode(adv.id, { parentId: 'gone', input: { kind: 'action', text: 'x' }, text: 'x' })).rejects.toThrow('is gone')
    await expect(store.addNode(adv.id, { parentId: root.node.id, input: { kind: 'start' }, text: 'x' })).rejects.toThrow('Only the opening starts')
  })
})

describe('state', () => {
  it('computes each node\'s state from its parent and carries it down', async () => {
    const t = await tree()
    const adv = await t.store.get(t.id)
    expect(adv.nodes[t.rootId].state).toEqual({ inventory: ['lamp'], stats: { Health: 10 }, flags: ['at the gate'] })
    // Stat names are matched without regard to case and keep the spelling they have.
    expect(adv.nodes[t.aId].state).toEqual({ inventory: ['lamp', 'key'], stats: { Health: 9 }, flags: ['at the gate'] })
    expect(adv.nodes[t.grandId].state).toEqual({ inventory: ['key'], stats: { Health: 9 }, flags: ['at the gate'] })
    expect(adv.nodes[t.bId].state).toEqual(adv.nodes[t.rootId].state)
  })

  it('a state edit stores the difference as changes, carries it down and raises every descendant\'s rev', async () => {
    const t = await tree()
    const before = await t.store.get(t.id)
    const root = before.nodes[t.rootId]
    const ch = await t.store.edit({
      adventureId: t.id,
      nodeId: t.rootId,
      rev: root.rev,
      state: { inventory: ['lamp', 'rope'], stats: { Health: 10, Gold: 5 }, flags: ['at the gate'] }
    })
    expect(ch.nodes?.map((n) => n.id).sort()).toEqual([t.rootId, t.aId, t.bId, t.grandId].sort())
    const after = await t.store.get(t.id)
    expect(after.nodes[t.rootId]).toMatchObject({ edited: true, rev: root.rev + 1 })
    expect(after.nodes[t.rootId].changes).toEqual({ ...noChanges(), addItems: ['lamp', 'rope'], setStats: [{ name: 'Health', value: 10 }, { name: 'Gold', value: 5 }], addFlags: ['at the gate'] })
    expect(after.nodes[t.aId].state.inventory).toEqual(['lamp', 'rope', 'key'])
    expect(after.nodes[t.aId].state.stats).toEqual({ Health: 9, Gold: 5 })
    expect(after.nodes[t.grandId].state.inventory).toEqual(['rope', 'key'])
    expect(after.nodes[t.bId].state.stats).toEqual({ Health: 10, Gold: 5 })
    for (const id of [t.aId, t.bId, t.grandId]) {
      expect(after.nodes[id].rev).toBe(before.nodes[id].rev + 1)
      // Carried down, not hand-edited.
      expect(after.nodes[id].edited).toBe(false)
    }
    // State = parent state + own changes, everywhere.
    expect(ch.meta?.nodeCount).toBe(4)
  })

  it('a text-only edit touches only that node; new choices replace the old ids', async () => {
    const t = await tree()
    const before = await t.store.get(t.id)
    const a = before.nodes[t.aId]
    const ch = await t.store.edit({ adventureId: t.id, nodeId: t.aId, rev: a.rev, text: '  A new voice.  ', choices: [{ id: a.choices[0].id, text: 'Enter slowly' }, { text: 'Brand new' }, { id: 'unknown', text: 'Other' }, { text: '  ' }] })
    expect(ch.nodes).toHaveLength(1)
    const now = (await t.store.get(t.id)).nodes[t.aId]
    expect(now.text).toBe('A new voice.')
    expect(now.choices.map((c) => c.text)).toEqual(['Enter slowly', 'Brand new', 'Other'])
    expect(now.choices[0].id).toBe(a.choices[0].id)
    expect(now.choices[1].id).not.toBe(a.choices[1].id)
    expect(now.choices[2].id).not.toBe('unknown')
    expect(now.rev).toBe(a.rev + 1)
    expect((await t.store.get(t.id)).nodes[t.grandId].rev).toBe(before.nodes[t.grandId].rev)
  })

  it('new choices (the model\'s reply) recompute the node and carry the change down the branch', async () => {
    const t = await tree()
    const before = await t.store.get(t.id)
    const ch = await t.store.setChoosing(t.id, t.aId, { choices: ['One', 'Two'], ended: false, changes: { ...noChanges(), addItems: ['sword'] } })
    expect(ch?.nodes?.map((n) => n.id)).toEqual([t.aId, t.grandId])
    const after = await t.store.get(t.id)
    expect(after.nodes[t.aId].state.inventory).toEqual(['lamp', 'sword'])
    expect(after.nodes[t.grandId].state.inventory).toEqual(['sword'])
    expect(after.nodes[t.aId].rev).toBe(before.nodes[t.aId].rev + 1)
    expect(after.nodes[t.grandId].rev).toBe(before.nodes[t.grandId].rev + 1)
    const ended = await t.store.setChoosing(t.id, t.bId, { choices: ['ignored'], ended: true, changes: noChanges() })
    expect(ended?.meta).toBeDefined()
    expect((await t.store.get(t.id)).nodes[t.bId]).toMatchObject({ ended: true, choices: [] })
  })
})

describe('choosing against a moved node, and a file under the wrong name', () => {
  it('setChoosing with the rev the request was built on saves nothing when the node moved on', async () => {
    const t = await tree()
    const a = (await t.store.get(t.id)).nodes[t.aId]
    await t.store.edit({ adventureId: t.id, nodeId: t.aId, rev: a.rev, text: 'Hand edited.' })
    const before = await t.store.get(t.id)
    t.changes.length = 0
    expect(await t.store.setChoosing(t.id, t.aId, { choices: ['Late'], ended: false, changes: noChanges() }, a.rev)).toBeNull()
    expect(t.changes).toEqual([])
    expect(await t.store.get(t.id)).toEqual(before)
    // With the current rev it is saved.
    const ok = await t.store.setChoosing(t.id, t.aId, { choices: ['Fresh'], ended: false, changes: noChanges() }, before.nodes[t.aId].rev)
    expect(ok?.nodes?.[0].choices.map((c) => c.text)).toEqual(['Fresh'])
    expect((await t.store.get(t.id)).nodes[t.aId].text).toBe('Hand edited.')
  })

  it('treats a file whose embedded id differs from its name as corrupt: set aside, not listed, not opened, not written', async () => {
    const { store, dir } = make()
    const one = await store.create(setup({ title: 'One' }))
    const two = await store.create(setup({ title: 'Two' }))
    // Two's content under One's name.
    writeFileSync(join(dir, `${one.id}.json`), JSON.stringify(two))
    expect((await store.list()).map((m) => m.id)).toEqual([two.id])
    expect(readdirSync(dir).some((n) => n.startsWith(`${one.id}.json.corrupt-`))).toBe(true)
    await expect(store.get(one.id)).rejects.toThrow('not found')
    await expect(store.updateSetup(one.id, { title: 'x' })).rejects.toThrow('not found')
    expect((await store.get(two.id)).setup.title).toBe('Two')
    expect(readdirSync(dir).filter((n) => n.endsWith('.json')).sort()).toEqual([`${two.id}.json`])
  })

  it('treats a file with a broken node or root as corrupt so a commit cannot throw half-way', async () => {
    const { store, dir } = make()
    const adv = await store.create(setup())
    writeFileSync(join(dir, `${adv.id}.json`), JSON.stringify({ ...adv, nodes: { x: { id: 'x', text: 1 } } }))
    await expect(store.get(adv.id)).rejects.toThrow('not found')
    const adv2 = await store.create(setup())
    writeFileSync(join(dir, `${adv2.id}.json`), JSON.stringify({ ...adv2, rootId: 5 }))
    expect(await store.list()).toEqual([])
  })
})

describe('conflicts', () => {
  it('refuses an edit based on an old rev, and an empty passage', async () => {
    const t = await tree()
    const a = (await t.store.get(t.id)).nodes[t.aId]
    await t.store.edit({ adventureId: t.id, nodeId: t.aId, rev: a.rev, text: 'Mine.' })
    await expect(t.store.edit({ adventureId: t.id, nodeId: t.aId, rev: a.rev, text: 'Theirs.' })).rejects.toThrow(CONFLICT_MESSAGE)
    await expect(t.store.edit({ adventureId: t.id, nodeId: t.aId, rev: a.rev + 1, text: '   ' })).rejects.toThrow('cannot be empty')
    await expect(t.store.edit({ adventureId: t.id, nodeId: 'nope', rev: 1, text: 'x' })).rejects.toThrow('not found')
    expect((await t.store.get(t.id)).nodes[t.aId].text).toBe('Mine.')
  })

  it('setCurrent raises no node rev, so it never causes a conflict', async () => {
    const t = await tree()
    const a = (await t.store.get(t.id)).nodes[t.aId]
    await t.store.setCurrent(t.id, t.aId)
    await t.store.setCurrent(t.id, t.bId)
    expect((await t.store.get(t.id)).nodes[t.aId].rev).toBe(a.rev)
    await t.store.edit({ adventureId: t.id, nodeId: t.aId, rev: a.rev, text: 'Still works.' })
  })
})

describe('deleteBranch and delete', () => {
  it('removes a node and everything under it and moves the reopen position to the parent', async () => {
    const t = await tree()
    await t.store.setCurrent(t.id, t.grandId)
    t.changes.length = 0
    const a = (await t.store.get(t.id)).nodes[t.aId]
    const ch = await t.store.deleteBranch(t.id, t.aId, a.rev)
    expect(ch.removedIds?.sort()).toEqual([t.aId, t.grandId].sort())
    expect(ch.currentId).toBe(t.rootId)
    expect(ch.nodes).toBeUndefined()
    expect(t.changes).toEqual([ch])
    const after = await t.store.get(t.id)
    expect(Object.keys(after.nodes).sort()).toEqual([t.rootId, t.bId].sort())
    expect(after.currentId).toBe(t.rootId)
    expect(ch.meta?.nodeCount).toBe(2)
  })

  it('leaves the reopen position alone when it is outside the branch', async () => {
    const t = await tree()
    await t.store.setCurrent(t.id, t.bId)
    const a = (await t.store.get(t.id)).nodes[t.aId]
    const ch = await t.store.deleteBranch(t.id, t.aId, a.rev)
    expect(ch.currentId).toBeUndefined()
    expect((await t.store.get(t.id)).currentId).toBe(t.bId)
  })

  it('refuses the root, a stale rev, an unknown node and a branch a job uses', async () => {
    const t = await tree()
    const adv = await t.store.get(t.id)
    await expect(t.store.deleteBranch(t.id, t.rootId, adv.nodes[t.rootId].rev)).rejects.toThrow('opening cannot be deleted')
    await expect(t.store.deleteBranch(t.id, t.aId, adv.nodes[t.aId].rev - 1)).rejects.toThrow(CONFLICT_MESSAGE)
    await expect(t.store.deleteBranch(t.id, 'nope', 1)).rejects.toThrow('not found')
    let asked: Set<string> | null = null
    await expect(
      t.store.deleteBranch(t.id, t.aId, adv.nodes[t.aId].rev, (removed) => {
        asked = removed
        return removed.has(t.grandId)
      })
    ).rejects.toThrow('being written')
    expect([...(asked as unknown as Set<string>)].sort()).toEqual([t.aId, t.grandId].sort())
    expect(Object.keys((await t.store.get(t.id)).nodes)).toHaveLength(4)
  })

  it('sends the file to the Recycle Bin and announces the delete', async () => {
    const { store, dir, trashed, changes } = make()
    const adv = await store.create(setup())
    changes.length = 0
    await store.delete(adv.id)
    expect(trashed).toEqual([join(dir, `${adv.id}.json`)])
    expect(existsSync(join(dir, `${adv.id}.json`))).toBe(false)
    expect(changes).toEqual([{ id: adv.id, rev: null }])
    await expect(store.delete(adv.id)).rejects.toThrow('not found')
    expect(await store.list()).toEqual([])
  })
})
