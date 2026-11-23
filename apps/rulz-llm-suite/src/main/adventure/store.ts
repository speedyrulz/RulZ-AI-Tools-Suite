// Saved adventures: one JSON file per adventure in `dir` (<userData>/adventures). No Electron imports: the app passes
// in the folder, the Recycle Bin function and the change listener, so tests run it on a temp folder.
// Design: docs/adventure-mode-design.md ("Saved files", "A turn").
import { promises as fs } from 'node:fs'
import { join } from 'node:path'
import {
  ADVENTURE_FORMAT,
  MAX_CHOICES,
  MIN_CHOICES,
  applyStateChanges,
  descendantsOf,
  diffState,
  emptyState,
  metaOf,
  noChanges,
  type Adventure,
  type AdventureChange,
  type AdventureEdit,
  type AdventureMeta,
  type AdventureNode,
  type AdventureSetup,
  type AdventureState,
  type AdventureWorld,
  type Choice,
  type NodeInput,
  type PassageLength,
  type StateChanges,
  type StatValue
} from '@shared/adventure'
import { corruptName, readJson, writeJsonAtomic } from '../util/jsonFile'

export const CONFLICT_MESSAGE = 'This passage was changed somewhere else. Reload it and try again.'
export const NEWER_FORMAT_MESSAGE = 'This adventure was saved by a newer version of RulZ LLM Suite. Update the app to open it.'

/** Ids are random; anything else is refused before a path is built from it. */
export const ADVENTURE_ID = /^[a-z0-9-]{8,64}$/i

export function checkAdventureId(id: unknown): string {
  if (typeof id !== 'string' || !ADVENTURE_ID.test(id)) throw new Error('That is not an adventure id.')
  return id
}

export interface AdventureStoreDeps {
  /** The folder the adventure files live in; created on first write. */
  dir: string
  now?: () => number
  /** Ids for adventures, nodes and choices; default crypto.randomUUID(). */
  newId?: () => string
  /** Sends a file to the Recycle Bin (shell.trashItem); without it a delete removes the file. */
  trash?: (path: string) => Promise<void>
  /** Called with every change, after it is saved. */
  emit?: (change: AdventureChange) => void
}

const LIMITS = { title: 120, genre: 80, premise: 4000, character: 1000, tone: 200, choice: 300 }
const LENGTHS: PassageLength[] = ['short', 'medium', 'long']

const clip = (v: unknown, max: number): string => (typeof v === 'string' ? v.trim().slice(0, max) : '')

const nodeAt = (adv: Adventure, id: unknown): AdventureNode | undefined =>
  typeof id === 'string' && Object.hasOwn(adv.nodes, id) ? adv.nodes[id] : undefined

/** The setup as stored: strings trimmed and capped, numbers clamped. `world` is set by main alone, never taken from a client. */
export function normalizeSetup(raw: Partial<AdventureSetup> | null | undefined, world?: AdventureWorld): AdventureSetup {
  const s = raw ?? {}
  const choiceCount = Math.min(MAX_CHOICES, Math.max(MIN_CHOICES, Math.round(Number(s.choiceCount)) || 3))
  const temp = typeof s.temperature === 'number' && Number.isFinite(s.temperature) ? Math.min(2, Math.max(0, s.temperature)) : null
  const out: AdventureSetup = {
    title: clip(s.title, LIMITS.title) || 'Untitled adventure',
    genre: clip(s.genre, LIMITS.genre),
    premise: clip(s.premise, LIMITS.premise),
    character: clip(s.character, LIMITS.character),
    tone: clip(s.tone, LIMITS.tone),
    length: LENGTHS.includes(s.length as PassageLength) ? (s.length as PassageLength) : 'medium',
    choiceCount,
    modelId: typeof s.modelId === 'string' && s.modelId ? s.modelId : null,
    temperature: temp
  }
  if (world) out.world = world
  return out
}

function normalizeState(raw: unknown): AdventureState {
  const o = (typeof raw === 'object' && raw !== null ? raw : {}) as Record<string, unknown>
  const list = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string').map((x) => x.trim()).filter(Boolean) : [])
  const stats: Record<string, StatValue> = {}
  if (typeof o.stats === 'object' && o.stats !== null && !Array.isArray(o.stats)) {
    for (const [k, v] of Object.entries(o.stats)) {
      if (!k.trim()) continue
      if (typeof v === 'number' && Number.isFinite(v)) stats[k.trim()] = v
      else if (typeof v === 'string') stats[k.trim()] = v
    }
  }
  return { inventory: list(o.inventory), stats, flags: list(o.flags) }
}

const isRec = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)

/** Enough of the shape that a commit cannot throw half-way; the file's own id must be the one asked for (its file name). */
function validShape(v: unknown, id: string): v is Adventure {
  if (!isRec(v)) return false
  if (v.id !== id || typeof v.format !== 'number' || typeof v.rev !== 'number' || typeof v.createdAt !== 'number' || typeof v.updatedAt !== 'number') return false
  if (!isRec(v.setup) || !isRec(v.nodes)) return false
  if (v.rootId !== null && typeof v.rootId !== 'string') return false
  if (v.currentId !== null && typeof v.currentId !== 'string') return false
  for (const [key, n] of Object.entries(v.nodes)) {
    if (!isRec(n) || n.id !== key || (n.parentId !== null && typeof n.parentId !== 'string')) return false
    if (typeof n.text !== 'string' || typeof n.rev !== 'number' || typeof n.createdAt !== 'number' || !Array.isArray(n.choices) || !isRec(n.input) || !isRec(n.changes) || !isRec(n.state)) return false
    if (!Array.isArray(n.state.inventory) || !Array.isArray(n.state.flags) || !isRec(n.state.stats)) return false
  }
  return true
}

export interface NewNode {
  parentId: string | null
  input: NodeInput
  text: string
}

export interface ChoosingResult {
  choices: string[]
  ended: boolean
  changes: StateChanges
}

export class AdventureStore {
  private readonly now: () => number
  private readonly newId: () => string
  /** Writes to one adventure run one after another. */
  private chains = new Map<string, Promise<unknown>>()

  constructor(private readonly deps: AdventureStoreDeps) {
    this.now = deps.now ?? Date.now
    this.newId = deps.newId ?? (() => globalThis.crypto.randomUUID())
  }

  /** A new adventure id (for callers that need it before create, e.g. to name the owner of a project they read). */
  makeId(): string {
    return this.newId()
  }

  private path(id: string): string {
    return join(this.deps.dir, `${checkAdventureId(id)}.json`)
  }

  // -------------------------------------------------------------------------------------------
  // Reading
  // -------------------------------------------------------------------------------------------

  /** The file as saved; null when it does not exist. A file that does not parse (or has no adventure shape) is set aside. */
  private async read(id: string): Promise<Adventure | null> {
    const path = this.path(id)
    const v = await readJson<unknown>(path, null)
    if (v === null) return null
    if (!validShape(v, id)) {
      await fs.rename(path, corruptName(path)).catch(() => undefined)
      console.warn(`${path} did not hold an adventure; kept it as ${corruptName(path)}.`)
      return null
    }
    return v
  }

  /** Every adventure, most recently played first. Corrupt files and files named like no adventure are left out. */
  async list(): Promise<AdventureMeta[]> {
    const names = await fs.readdir(this.deps.dir).catch(() => [] as string[])
    const out: AdventureMeta[] = []
    for (const name of names) {
      const m = /^(.+)\.json$/.exec(name)
      if (!m || !ADVENTURE_ID.test(m[1])) continue
      const adv = await this.read(m[1]).catch(() => null)
      if (adv) out.push(metaOf(adv))
    }
    return out.sort((a, b) => b.updatedAt - a.updatedAt || a.id.localeCompare(b.id))
  }

  /** One adventure; refused when a newer version of the app wrote it (so nothing can overwrite it either). */
  async get(id: string): Promise<Adventure> {
    const adv = await this.read(id)
    if (!adv) throw new Error('That adventure was not found.')
    if (adv.format > ADVENTURE_FORMAT) throw new Error(NEWER_FORMAT_MESSAGE)
    return adv
  }

  // -------------------------------------------------------------------------------------------
  // Writing
  // -------------------------------------------------------------------------------------------

  /** Runs `fn` after the earlier writes to this adventure. */
  private serial<T>(id: string, fn: () => Promise<T>): Promise<T> {
    const prev = this.chains.get(id) ?? Promise.resolve()
    const next = prev.catch(() => undefined).then(fn)
    this.chains.set(id, next)
    const cleanup = (): void => {
      if (this.chains.get(id) === next) this.chains.delete(id)
    }
    void next.then(cleanup, cleanup)
    return next
  }

  /** Raises the rev, saves, and announces the change (meta always). Returns the change so callers hand back the same object. */
  private async commit(adv: Adventure, change: Omit<AdventureChange, 'id' | 'rev' | 'meta'>): Promise<AdventureChange> {
    adv.rev += 1
    adv.updatedAt = Math.max(this.now(), adv.updatedAt)
    await writeJsonAtomic(this.path(adv.id), adv)
    const out: AdventureChange = { id: adv.id, rev: adv.rev, ...change, meta: metaOf(adv) }
    this.deps.emit?.(out)
    return out
  }

  /** Creates an adventure with no passages yet. `world` comes from main (a Writing project's digest); a world inside `setup` is ignored. */
  async create(setup: Partial<AdventureSetup>, world?: AdventureWorld, id?: string): Promise<Adventure> {
    const newId = id ? checkAdventureId(id) : this.newId()
    checkAdventureId(newId)
    return this.serial(newId, async () => {
      if (await this.read(newId)) throw new Error('That adventure already exists.')
      const t = this.now()
      const adv: Adventure = {
        format: ADVENTURE_FORMAT,
        id: newId,
        setup: normalizeSetup(setup, world),
        nodes: {},
        rootId: null,
        currentId: null,
        rev: 0,
        createdAt: t,
        updatedAt: t
      }
      await this.commit(adv, { setup: adv.setup })
      return adv
    })
  }

  /** Title and generation settings; the premise, character and world stay. Last write wins. */
  updateSetup(id: string, patch: Partial<AdventureSetup>): Promise<AdventureChange> {
    checkAdventureId(id)
    return this.serial(id, async () => {
      const adv = await this.get(id)
      const p = { ...(patch ?? {}) } as Record<string, unknown>
      // A value that is not usable (an unknown length, a count that is not a number) leaves the setting as it was.
      if (!LENGTHS.includes(p.length as PassageLength)) delete p.length
      if (typeof p.choiceCount !== 'number') delete p.choiceCount
      const next = normalizeSetup({ ...adv.setup, ...pick(p, ['title', 'genre', 'tone', 'length', 'choiceCount', 'modelId', 'temperature']) }, adv.setup.world)
      // normalizeSetup falls back to defaults for what it cannot use: a title cleared by the patch keeps the old one.
      if (!clip(p.title, LIMITS.title)) next.title = adv.setup.title
      adv.setup = { ...next, premise: adv.setup.premise, character: adv.setup.character }
      return this.commit(adv, { setup: adv.setup })
    })
  }

  /** Where to reopen. Raises no node's rev. */
  setCurrent(id: string, nodeId: string): Promise<AdventureChange> {
    checkAdventureId(id)
    return this.serial(id, async () => {
      const adv = await this.get(id)
      if (!nodeAt(adv, nodeId)) throw new Error('That passage was not found.')
      adv.currentId = nodeId
      return this.commit(adv, { currentId: nodeId })
    })
  }

  /** Saves a new passage (no choices yet) under `parentId` and makes it the reopen position. */
  addNode(id: string, n: NewNode): Promise<{ change: AdventureChange; node: AdventureNode }> {
    checkAdventureId(id)
    return this.serial(id, async () => {
      const adv = await this.get(id)
      let parentState = emptyState()
      if (n.parentId === null) {
        if (adv.rootId !== null) throw new Error('The opening has already been written.')
        if (n.input.kind !== 'start') throw new Error('Only the opening has no parent.')
      } else {
        const parent = nodeAt(adv, n.parentId)
        if (!parent) throw new Error('The passage this continues from is gone.')
        if (n.input.kind === 'start') throw new Error('Only the opening starts the story.')
        parentState = parent.state
      }
      const nodeId = this.newId()
      const node: AdventureNode = {
        id: nodeId,
        parentId: n.parentId,
        input: n.input,
        text: n.text,
        choices: [],
        ended: false,
        changes: noChanges(),
        state: { inventory: [...parentState.inventory], stats: { ...parentState.stats }, flags: [...parentState.flags] },
        edited: false,
        rev: 1,
        createdAt: this.now()
      }
      adv.nodes[nodeId] = node
      const change: Omit<AdventureChange, 'id' | 'rev' | 'meta'> = { nodes: [node], currentId: nodeId }
      adv.currentId = nodeId
      if (n.parentId === null) {
        adv.rootId = nodeId
        change.rootId = nodeId
      }
      return { change: await this.commit(adv, change), node }
    })
  }

  /**
   * Sets a node's choices, ended and changes (the model's reply), recomputes its state and carries it down the branch.
   * With `baseRev`, nothing is saved (null) when the node changed since the reply's request was built: a hand edit
   * made meanwhile must not be overwritten by choices written from the old text.
   */
  setChoosing(id: string, nodeId: string, r: ChoosingResult, baseRev?: number): Promise<AdventureChange | null> {
    checkAdventureId(id)
    return this.serial(id, async () => {
      const adv = await this.get(id)
      const node = nodeAt(adv, nodeId)
      if (!node) throw new Error('That passage was not found.')
      if (baseRev !== undefined && node.rev !== baseRev) return null
      node.ended = r.ended
      node.choices = r.ended ? [] : r.choices.map((text) => ({ id: this.newId(), text }))
      node.changes = r.changes
      node.rev += 1
      const touched = this.carryDown(adv, node)
      return this.commit(adv, { nodes: touched })
    })
  }

  /** The node's state from its parent's and its own changes, then every node below it, parents first. Returns all of them (rev raised). */
  private carryDown(adv: Adventure, node: AdventureNode): AdventureNode[] {
    const parent = node.parentId ? nodeAt(adv, node.parentId) : undefined
    node.state = applyStateChanges(parent?.state ?? emptyState(), node.changes)
    const touched = [node]
    for (const cid of descendantsOf(adv, node.id)) {
      const c = adv.nodes[cid]
      c.state = applyStateChanges(adv.nodes[c.parentId as string].state, c.changes)
      c.rev += 1
      touched.push(c)
    }
    return touched
  }

  /** A hand edit of one node, refused when the node moved on since the editor read it. */
  edit(e: AdventureEdit): Promise<AdventureChange> {
    checkAdventureId(e?.adventureId)
    return this.serial(e.adventureId, async () => {
      const adv = await this.get(e.adventureId)
      const node = nodeAt(adv, e.nodeId)
      if (!node) throw new Error('That passage was not found.')
      if (e.rev !== node.rev) throw new Error(CONFLICT_MESSAGE)
      let stateChanged = false
      if (e.text !== undefined) {
        if (typeof e.text !== 'string' || !e.text.trim()) throw new Error('A passage cannot be empty.')
        node.text = e.text.trim()
      }
      if (e.choices !== undefined) {
        if (!Array.isArray(e.choices)) throw new Error('The choices must be a list.')
        const known = new Set(node.choices.map((c) => c.id))
        const seen = new Set<string>()
        const next: Choice[] = []
        for (const c of e.choices.slice(0, 10)) {
          const text = clip(c?.text, LIMITS.choice)
          if (!text) continue
          const keep = typeof c.id === 'string' && known.has(c.id) && !seen.has(c.id)
          const cid = keep ? (c.id as string) : this.newId()
          seen.add(cid)
          next.push({ id: cid, text })
        }
        node.choices = next
        // Choices on a passage that ended the story mean the reader wants it to go on.
        if (next.length) node.ended = false
      }
      if (e.state !== undefined) {
        const parent = node.parentId ? nodeAt(adv, node.parentId) : undefined
        const base = parent?.state ?? emptyState()
        node.changes = diffState(base, normalizeState(e.state))
        stateChanged = true
      }
      node.edited = true
      node.rev += 1
      const touched = stateChanged ? this.carryDown(adv, node) : [node]
      return this.commit(adv, { nodes: touched })
    })
  }

  /**
   * Removes a node and everything under it. Refused for the root, when `rev` is stale, and when `inUse` says a job
   * works on the branch. A reopen position inside the branch moves to the node's parent.
   */
  deleteBranch(id: string, nodeId: string, rev: number, inUse?: (removed: Set<string>) => boolean): Promise<AdventureChange> {
    checkAdventureId(id)
    return this.serial(id, async () => {
      const adv = await this.get(id)
      const node = nodeAt(adv, nodeId)
      if (!node) throw new Error('That passage was not found.')
      if (node.parentId === null) throw new Error('The opening cannot be deleted. Delete the adventure instead.')
      if (rev !== node.rev) throw new Error(CONFLICT_MESSAGE)
      const removed = [node.id, ...descendantsOf(adv, node.id)]
      if (inUse?.(new Set(removed))) throw new Error('A passage in this branch is being written. Stop it first.')
      for (const rid of removed) delete adv.nodes[rid]
      const change: Omit<AdventureChange, 'id' | 'rev' | 'meta'> = { removedIds: removed }
      if (adv.currentId !== null && removed.includes(adv.currentId)) {
        adv.currentId = node.parentId
        change.currentId = adv.currentId
      }
      return this.commit(adv, change)
    })
  }

  /** Sends the file to the Recycle Bin and announces the delete. */
  delete(id: string): Promise<void> {
    checkAdventureId(id)
    return this.serial(id, async () => {
      const path = this.path(id)
      if (!(await fs.stat(path).catch(() => null))) throw new Error('That adventure was not found.')
      if (this.deps.trash) await this.deps.trash(path)
      else await fs.rm(path, { force: true })
      this.deps.emit?.({ id, rev: null })
    })
  }
}

function pick(o: Record<string, unknown>, keys: string[]): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const k of keys) if (k in o && o[k] !== undefined) out[k] = o[k]
  return out
}
