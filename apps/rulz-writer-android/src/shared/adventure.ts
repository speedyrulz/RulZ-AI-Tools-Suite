// Adventure mode: choose-your-own-adventure stories played with the loaded model.
// Design: docs/adventure-mode-design.md. This file is the contract between main, the desktop window and the phone
// (RulZ Writer for Android copies it verbatim with its sync-shared script), so it must not import anything and must
// stay free of node: imports.
//
// An adventure is a tree of passages. The root is the opening; every other node is what happened after the reader
// picked one of its parent's choices or typed their own action. Going back and picking something else adds a
// sibling, so no path is ever lost. Generation runs in main as jobs (like the phone's Writing runs), so the window
// and the phone start, watch and stop the same jobs through the same channels.

/** Bumped when the saved file's shape changes in a way older code cannot read. */
export const ADVENTURE_FORMAT = 1

// ---------------------------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------------------------

export type PassageLength = 'short' | 'medium' | 'long'

/** Rough word targets per passage length, used in the prompt and to size the reply budget. */
export const PASSAGE_WORDS: Record<PassageLength, number> = { short: 120, medium: 250, long: 450 }

export const MIN_CHOICES = 2
export const MAX_CHOICES = 4

/** The setting taken from a Writing project when the adventure was created. Copied in, so the adventure keeps
 *  working if the project changes, moves or is deleted. */
export interface AdventureWorld {
  /** The project's folder and title when it was copied (shown only; never opened again by the adventure). */
  folder: string
  title: string
  /** Characters, places and other story-bible entries, plus the project's premise, as plain text for the prompt. */
  text: string
}

export interface AdventureSetup {
  title: string
  /** Free text, e.g. "Fantasy", "Space opera", "Noir mystery". */
  genre: string
  /** What the story is about and where it starts. May be empty when `world` is given (the project's premise is used). */
  premise: string
  /** Who the reader plays: name and a line about them. May be empty (the model invents one). */
  character: string
  /** Free text, e.g. "Light and funny", "Grim", "Cosy". */
  tone: string
  length: PassageLength
  /** Choices offered after each passage, MIN_CHOICES..MAX_CHOICES. */
  choiceCount: number
  /** Present when the adventure was started from a Writing project. Set by main from adventure:create's worldFolder;
   *  a world sent in `setup` by a client is ignored. */
  world?: AdventureWorld
  /** Model library id to play with; null = whatever model is loaded (the desktop's active model). */
  modelId: string | null
  /** Sampling temperature for passages; null = the app's default sampling (DEFAULT_SAMPLING in src/shared/defaults.ts). */
  temperature: number | null
}

/** A ready-made premise that fills the setup form. */
export interface AdventureStarter {
  id: string
  title: string
  genre: string
  premise: string
  character: string
  tone: string
}

// ---------------------------------------------------------------------------------------------
// Tracked state
// ---------------------------------------------------------------------------------------------

export type StatValue = number | string

/** What the story must stay consistent with. Each node keeps the state as it stood after that passage, so going
 *  back to an earlier passage also goes back to its inventory and stats. */
export interface AdventureState {
  /** Items the character carries, as short names ("rusty key", "3 gold coins"). */
  inventory: string[]
  /** Named values ("Health": 8, "Gold": 12, "Mood": "wary"). Names are matched without regard to case. */
  stats: Record<string, StatValue>
  /** Facts that have become true ("met the ferryman", "bridge is burned"). */
  flags: string[]
}

export function emptyState(): AdventureState {
  return { inventory: [], stats: {}, flags: [] }
}

/** Changes after a passage: proposed by the model, or the difference a hand edit made. Kept on the node, so a change
 *  higher up the branch can be carried down by applying each descendant's changes again. */
export interface StateChanges {
  addItems: string[]
  removeItems: string[]
  /** Set or overwrite a stat (a list, not an object: schema-constrained output handles fixed keys far better). */
  setStats: { name: string; value: StatValue }[]
  /** Stat names to remove. */
  removeStats: string[]
  addFlags: string[]
  removeFlags: string[]
}

export function noChanges(): StateChanges {
  return { addItems: [], removeItems: [], setStats: [], removeStats: [], addFlags: [], removeFlags: [] }
}

// ---------------------------------------------------------------------------------------------
// The tree
// ---------------------------------------------------------------------------------------------

/** One offered choice. The id stays with the choice for good: New choices or an edit makes new ids, so a child made
 *  from an old choice is never mistaken for one made from a new choice at the same position. */
export interface Choice {
  id: string
  text: string
}

/** How the reader got to a node: the opening, one of the parent's offered choices, or their own typed action. */
export type NodeInput =
  | { kind: 'start' }
  /** `text` is the choice's text when it was picked (kept even if the parent's choices change later). */
  | { kind: 'choice'; choiceId: string; text: string }
  | { kind: 'action'; text: string }

export interface AdventureNode {
  id: string
  /** null only for the root (the opening). */
  parentId: string | null
  input: NodeInput
  /** The passage. */
  text: string
  /** The choices offered after this passage. Empty when the story ended here, and also while (or when) choosing did
   *  not finish: then `ended` is false and the reader can ask for New choices or type an action. */
  choices: Choice[]
  /** The model said the story ends here (death, victory, a natural close). The reader can still go back, or type an
   *  action to keep going. */
  ended: boolean
  /** What this passage changed: applied to the parent's state (the root: to emptyState()) it gives `state`. */
  changes: StateChanges
  /** State after this passage (kept so readers need no replay; main keeps it equal to parent state + changes). */
  state: AdventureState
  /** The reader edited the passage, choices or state by hand. */
  edited: boolean
  /** Raised by one on every change to this node; hand edits name the rev they are based on. */
  rev: number
  createdAt: number
}

/** The saved file: <userData>/adventures/<id>.json. */
export interface Adventure {
  format: number
  id: string
  setup: AdventureSetup
  /** Every node by id. Children are found by parentId (see childrenOf). */
  nodes: Record<string, AdventureNode>
  /** The root node's id, or null before the opening has been written. */
  rootId: string | null
  /** Where to reopen: the node the reader was at last, on any device. Each window or phone keeps its own position
   *  while it has the adventure open; this only moves on setCurrent, a finished play / retryPassage, or deleteBranch. */
  currentId: string | null
  /** Raised by one on every save (ordering of changes; not used to refuse edits). */
  rev: number
  createdAt: number
  updatedAt: number
}

/** One row of the adventure list. */
export interface AdventureMeta {
  id: string
  title: string
  genre: string
  /** Passages in the whole tree. */
  nodeCount: number
  /** Passages from the root to the current node. */
  depth: number
  /** The current node ended the story. */
  ended: boolean
  /** Title of the Writing project it came from, if any. */
  worldTitle?: string
  createdAt: number
  updatedAt: number
}

// ---------------------------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------------------------

/**
 * play:          write a new child of `nodeId` from `input` (nodeId null with input 'start' = the opening).
 * retryPassage:  write the same step again as a new sibling of `nodeId` (same parent, same input); the old one stays.
 * retryChoices:  write new choices, ended and changes for `nodeId` in place (children already made keep their own
 *                input text; the branch below has its state carried down again).
 *
 * play and retryPassage save the new node as soon as the passage is written (with no choices yet), then go on to
 * choosing as retryChoices does. So a stop or failure while choosing keeps the passage.
 */
export type AdventureJobKind = 'play' | 'retryPassage' | 'retryChoices'

export type AdventureJobStatus =
  | 'queued'
  /** Loading the model, reading the adventure, counting tokens. */
  | 'preparing'
  /** The model is reading the prompt (see progress). */
  | 'waiting'
  /** The passage is streaming. */
  | 'writing'
  /** The passage is saved; the choices and state changes are being written. */
  | 'choosing'
  | 'done'
  | 'stopped'
  | 'error'

export interface AdventureJob {
  id: string
  adventureId: string
  kind: AdventureJobKind
  /** play: the parent the new node goes under (null = the opening). retryPassage / retryChoices: the node retried. */
  nodeId: string | null
  input: NodeInput | null
  status: AdventureJobStatus
  /** Prompt reading progress 0..1 while waiting; null when unknown. */
  progress: number | null
  /** The passage as it streams (empty for retryChoices). */
  text: string
  /** The node that was written or changed; set as soon as a passage is saved (status 'choosing'), so a client can
   *  show it while the choices are written. */
  resultNodeId: string | null
  error: string | null
  /** Notes worth showing (older passages left out to fit the context, a reply that had to be repaired). */
  warnings: string[]
  startedAt: number
  endedAt: number | null
}

export interface PlayRequest {
  adventureId: string
  /** null with input 'start' writes the opening; refused when the adventure already has one. A 'choice' input must
   *  name a choice id that is on the parent now. An ended parent takes only an 'action' input. */
  parentId: string | null
  input: NodeInput
}

export interface RetryRequest {
  adventureId: string
  nodeId: string
  what: 'passage' | 'choices'
}

/** A hand edit of one node. Only the fields given change. Refused with a conflict error when `rev` is not the node's
 *  current rev (it changed somewhere else): reload it and try again.
 *  - choices: give ids for kept choices; a choice without an id (or with an unknown one) gets a new id.
 *  - state: stored as the node's new state; main stores the difference from the parent's state as `changes` and
 *    carries it down the branch below. */
export interface AdventureEdit {
  adventureId: string
  nodeId: string
  rev: number
  text?: string
  choices?: { id?: string; text: string }[]
  state?: AdventureState
}

/** What a save changed. Sent as the 'adventure:changed' event to the window and the phones, and returned by the
 *  calls that change an adventure, so clients patch what they hold instead of fetching the whole tree again. */
export interface AdventureChange {
  id: string
  /** The adventure's new rev; null when the adventure was deleted. */
  rev: number | null
  /** Nodes added or changed (whole nodes). */
  nodes?: AdventureNode[]
  /** Nodes removed (deleteBranch). */
  removedIds?: string[]
  /** Set when the saved reopen position moved. */
  currentId?: string | null
  /** Set when rootId changed (the opening was written). */
  rootId?: string | null
  /** Set when the setup changed. */
  setup?: AdventureSetup
  /** Set when the list row changed (always, except on delete). */
  meta?: AdventureMeta
}

/** Applies a change to an adventure a client holds; returns a new object (the old one is not modified). A change for
 *  another adventure or a delete returns the adventure unchanged; the caller handles those. */
export function applyChange(adv: Adventure, ch: AdventureChange): Adventure {
  if (ch.id !== adv.id || ch.rev === null) return adv
  const nodes = { ...adv.nodes }
  for (const id of ch.removedIds ?? []) delete nodes[id]
  for (const n of ch.nodes ?? []) nodes[n.id] = n
  return {
    ...adv,
    nodes,
    rev: Math.max(adv.rev, ch.rev),
    currentId: ch.currentId !== undefined ? ch.currentId : adv.currentId,
    rootId: ch.rootId !== undefined ? ch.rootId : adv.rootId,
    setup: ch.setup ?? adv.setup,
    updatedAt: ch.meta?.updatedAt ?? adv.updatedAt
  }
}

// ---------------------------------------------------------------------------------------------
// Channels (answered in main for the window and, through the phone dispatcher, for phones)
// ---------------------------------------------------------------------------------------------

export interface AdventureIpcInvoke {
  'adventure:list': () => AdventureMeta[]
  'adventure:get': (id: string) => Adventure
  /** Creates the adventure (no opening yet; start it with adventure:play { parentId: null, input: start }). When
   *  `worldFolder` is given, the Writing project's premise and story bible are copied into setup.world. */
  'adventure:create': (setup: AdventureSetup, worldFolder?: string | null) => Adventure
  /** Changes the title or the generation settings (genre, tone, length, choiceCount, modelId, temperature). The
   *  premise, character and world stay as they were. Last write wins. */
  'adventure:updateSetup': (id: string, patch: Partial<Omit<AdventureSetup, 'world' | 'premise' | 'character'>>) => AdventureChange
  /** Saves the reopen position. Does not raise any node's rev and does not move other clients' positions. */
  'adventure:setCurrent': (id: string, nodeId: string) => AdventureChange
  'adventure:edit': (edit: AdventureEdit) => AdventureChange
  /** The path from the root to `nodeId` (default: the current node) as one Markdown story. */
  'adventure:exportMarkdown': (id: string, nodeId?: string | null) => string
  /** Starts a job; refused while another model job runs anywhere on the computer. */
  'adventure:play': (req: PlayRequest) => AdventureJob
  'adventure:retry': (req: RetryRequest) => AdventureJob
  'adventure:jobs': (adventureId: string) => AdventureJob[]
  'adventure:abort': (jobId: string) => void
  /** Forgets a finished job. */
  'adventure:dismiss': (jobId: string) => void
  'adventure:starters': () => AdventureStarter[]
  /** The model library, for the setup's model picker. */
  'adventure:models': () => { id: string; name: string; loaded: boolean }[]
}

/** Desktop only (not answered for phones). */
export interface AdventureDesktopInvoke {
  /** Sends the adventure file to the Recycle Bin. */
  'adventure:delete': (id: string) => void
  /** Removes a node and everything under it, for good (asked first in the UI). Refused for the root, when `rev` is
   *  not the node's current rev, and while a job uses the branch. A reopen position inside the branch moves to the
   *  node's parent. */
  'adventure:deleteBranch': (id: string, nodeId: string, rev: number) => AdventureChange
  /** Asks where to save and writes the Markdown file; returns the path, or null when cancelled. */
  'adventure:saveExport': (id: string, nodeId?: string | null) => string | null
}

export interface AdventureIpcEvents {
  /** Every status change, progress change (throttled) and, throttled to about 4 per second, while the passage streams. */
  'adventure:job': AdventureJob
  'adventure:changed': AdventureChange
}

/** The channels a phone may call. */
export const PHONE_ADVENTURE_CHANNELS = [
  'adventure:list',
  'adventure:get',
  'adventure:create',
  'adventure:updateSetup',
  'adventure:setCurrent',
  'adventure:edit',
  'adventure:exportMarkdown',
  'adventure:play',
  'adventure:retry',
  'adventure:jobs',
  'adventure:abort',
  'adventure:dismiss',
  'adventure:starters',
  'adventure:models'
] as const satisfies readonly (keyof AdventureIpcInvoke)[]

// ---------------------------------------------------------------------------------------------
// Pure tree helpers (shared so every side reads the tree the same way)
// ---------------------------------------------------------------------------------------------

/** A node's children, oldest first. */
export function childrenOf(adv: Adventure, nodeId: string): AdventureNode[] {
  return Object.values(adv.nodes)
    .filter((n) => n.parentId === nodeId)
    .sort((a, b) => a.createdAt - b.createdAt || a.id.localeCompare(b.id))
}

/** The nodes from the root to `nodeId`, root first. Empty when the node is unknown. Stops on a cycle. */
export function pathTo(adv: Adventure, nodeId: string | null): AdventureNode[] {
  const out: AdventureNode[] = []
  const seen = new Set<string>()
  let n = nodeId ? adv.nodes[nodeId] : undefined
  while (n && !seen.has(n.id)) {
    seen.add(n.id)
    out.push(n)
    n = n.parentId ? adv.nodes[n.parentId] : undefined
  }
  return out.reverse()
}

/** The other ways the story went from the same parent (including `nodeId` itself), oldest first. */
export function siblingsOf(adv: Adventure, nodeId: string): AdventureNode[] {
  const n = adv.nodes[nodeId]
  if (!n) return []
  if (n.parentId === null) return [n]
  return childrenOf(adv, n.parentId)
}

/** Every node under `nodeId`, not including it, parents before children. */
export function descendantsOf(adv: Adventure, nodeId: string): string[] {
  const out: string[] = []
  const queue = [nodeId]
  const seen = new Set<string>([nodeId])
  while (queue.length) {
    const id = queue.shift() as string
    for (const c of childrenOf(adv, id)) {
      if (seen.has(c.id)) continue
      seen.add(c.id)
      out.push(c.id)
      queue.push(c.id)
    }
  }
  return out
}

/** The newest child made from choice `choiceId` of `nodeId` (several when it was retried). */
export function childForChoice(adv: Adventure, nodeId: string, choiceId: string): AdventureNode | undefined {
  const kids = childrenOf(adv, nodeId).filter((c) => c.input.kind === 'choice' && c.input.choiceId === choiceId)
  return kids[kids.length - 1]
}

/** The newest leaf under `nodeId` following the newest child at each step (where "switch to this path" lands). */
export function newestLeaf(adv: Adventure, nodeId: string): AdventureNode | undefined {
  let n = adv.nodes[nodeId]
  const seen = new Set<string>()
  while (n && !seen.has(n.id)) {
    seen.add(n.id)
    const kids = childrenOf(adv, n.id)
    if (!kids.length) return n
    n = kids[kids.length - 1]
  }
  return n
}

const key = (s: string): string => s.trim().toLowerCase()

export function applyStateChanges(base: AdventureState, ch: StateChanges): AdventureState {
  const removeItems = new Set(ch.removeItems.map(key))
  const removeFlags = new Set(ch.removeFlags.map(key))
  const inventory = base.inventory.filter((i) => !removeItems.has(key(i)))
  for (const i of ch.addItems) if (i.trim() && !inventory.some((x) => key(x) === key(i))) inventory.push(i.trim())
  const flags = base.flags.filter((f) => !removeFlags.has(key(f)))
  for (const f of ch.addFlags) if (f.trim() && !flags.some((x) => key(x) === key(f))) flags.push(f.trim())
  const stats: Record<string, StatValue> = {}
  const removeStats = new Set(ch.removeStats.map(key))
  for (const [k, v] of Object.entries(base.stats)) if (!removeStats.has(key(k))) stats[k] = v
  for (const { name, value } of ch.setStats) {
    if (!name.trim()) continue
    // Keep the spelling the stat already has ("Health" stays "Health" when the model writes "health").
    const existing = Object.keys(stats).find((k) => key(k) === key(name))
    stats[existing ?? name.trim()] = value
  }
  return { inventory, stats, flags }
}

/** The changes that turn `from` into `to` (used to store a hand-edited state as changes). */
export function diffState(from: AdventureState, to: AdventureState): StateChanges {
  const has = (list: string[], s: string): boolean => list.some((x) => key(x) === key(s))
  const fromStats = new Map(Object.entries(from.stats).map(([k, v]) => [key(k), v]))
  const toKeys = new Set(Object.keys(to.stats).map(key))
  return {
    addItems: to.inventory.filter((i) => !has(from.inventory, i)),
    removeItems: from.inventory.filter((i) => !has(to.inventory, i)),
    setStats: Object.entries(to.stats)
      .filter(([k, v]) => fromStats.get(key(k)) !== v)
      .map(([name, value]) => ({ name, value })),
    removeStats: Object.keys(from.stats).filter((k) => !toKeys.has(key(k))),
    addFlags: to.flags.filter((f) => !has(from.flags, f)),
    removeFlags: from.flags.filter((f) => !has(to.flags, f))
  }
}

export function metaOf(adv: Adventure): AdventureMeta {
  const current = adv.currentId ? adv.nodes[adv.currentId] : undefined
  return {
    id: adv.id,
    title: adv.setup.title,
    genre: adv.setup.genre,
    nodeCount: Object.keys(adv.nodes).length,
    depth: pathTo(adv, adv.currentId).length,
    ended: current?.ended ?? false,
    worldTitle: adv.setup.world?.title,
    createdAt: adv.createdAt,
    updatedAt: adv.updatedAt
  }
}
