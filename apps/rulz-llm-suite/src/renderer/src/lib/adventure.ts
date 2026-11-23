// Adventure mode, renderer side: pure helpers (no React, no IPC) so the store and the tests can use them.
import {
  childrenOf,
  type Adventure,
  type AdventureJob,
  type AdventureNode,
  type AdventureSetup,
  type AdventureStarter,
  type AdventureState,
  type NodeInput,
  type PassageLength,
  type StatValue
} from '@shared/adventure'

export const LENGTH_LABELS: Record<PassageLength, string> = { short: 'Short', medium: 'Medium', long: 'Long' }

/** A job that has not finished yet. */
export function isRunning(job: AdventureJob | null | undefined): boolean {
  return !!job && job.status !== 'done' && job.status !== 'stopped' && job.status !== 'error'
}

export function isFinished(job: AdventureJob): boolean {
  return !isRunning(job)
}

/** The refusal main sends when a node moved on since it was read. */
export function isConflict(message: string): boolean {
  return /changed somewhere else/i.test(message)
}

export function shorten(text: string, max: number): string {
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length > max ? `${flat.slice(0, Math.max(1, max - 1)).trimEnd()}…` : flat
}

/** The step the reader took to reach a node, as one muted line. */
export function stepLine(input: NodeInput): string {
  if (input.kind === 'choice') return `You chose: ${input.text}`
  if (input.kind === 'action') return `You: ${input.text}`
  return ''
}

/** Short label of a node's step for the map. */
export function mapLabel(node: AdventureNode, max = 44): string {
  if (node.input.kind === 'start') return 'The opening'
  const text = shorten(node.input.text, max)
  return node.input.kind === 'action' ? `You: ${text}` : text
}

export interface MapRow {
  node: AdventureNode
  depth: number
  onPath: boolean
  isPosition: boolean
}

/** The tree as an indented outline (children oldest first), with the path to `position` marked. */
export function mapRows(adv: Adventure, position: string | null): MapRow[] {
  if (!adv.rootId || !adv.nodes[adv.rootId]) return []
  const kids = new Map<string | null, AdventureNode[]>()
  for (const n of Object.values(adv.nodes)) {
    const list = kids.get(n.parentId) ?? []
    list.push(n)
    kids.set(n.parentId, list)
  }
  for (const list of kids.values()) list.sort((a, b) => a.createdAt - b.createdAt || a.id.localeCompare(b.id))
  const onPath = new Set<string>()
  const seen = new Set<string>()
  let p = position ? adv.nodes[position] : undefined
  while (p && !seen.has(p.id)) {
    seen.add(p.id)
    onPath.add(p.id)
    p = p.parentId ? adv.nodes[p.parentId] : undefined
  }
  const rows: MapRow[] = []
  const visited = new Set<string>()
  const stack: { node: AdventureNode; depth: number }[] = [{ node: adv.nodes[adv.rootId], depth: 0 }]
  while (stack.length) {
    const { node, depth } = stack.pop() as { node: AdventureNode; depth: number }
    if (visited.has(node.id)) continue
    visited.add(node.id)
    rows.push({ node, depth, onPath: onPath.has(node.id), isPosition: node.id === position })
    const children = kids.get(node.id) ?? []
    for (let i = children.length - 1; i >= 0; i--) stack.push({ node: children[i], depth: depth + 1 })
  }
  return rows
}

/** What the reading column shows while a job runs. */
export type LiveView =
  | { kind: 'none' }
  /** The passage is being written: show its text after the path (retryPassage: instead of `replaces`). */
  | { kind: 'passage'; job: AdventureJob; replaces: string | null }
  /** The passage is saved and shown; the choices are being written under it. */
  | { kind: 'choices'; job: AdventureJob }

export function liveView(job: AdventureJob | null, position: string | null): LiveView {
  if (!job || !isRunning(job)) return { kind: 'none' }
  if (job.kind === 'retryChoices') return job.nodeId === position ? { kind: 'choices', job } : { kind: 'none' }
  if (job.status === 'choosing') return job.resultNodeId && job.resultNodeId === position ? { kind: 'choices', job } : { kind: 'none' }
  if (job.nodeId !== position) return { kind: 'none' }
  return { kind: 'passage', job, replaces: job.kind === 'retryPassage' ? job.nodeId : null }
}

export function jobStatusText(job: AdventureJob): string {
  if (job.status === 'queued' || job.status === 'preparing') return 'Getting ready…'
  if (job.status === 'waiting') return 'Reading the story…'
  if (job.status === 'writing') return 'Writing…'
  if (job.status === 'choosing') return 'Writing choices…'
  return ''
}

export function stateIsEmpty(st: AdventureState): boolean {
  return st.inventory.length === 0 && st.flags.length === 0 && Object.keys(st.stats).length === 0
}

export function statText(v: StatValue): string {
  return String(v)
}

/** A typed stat value: a number when the text is one, else the text. */
export function parseStat(text: string): StatValue {
  const t = text.trim()
  if (t !== '' && Number.isFinite(Number(t))) return Number(t)
  return t
}

/** A list moved from one position to another (a new array). */
export function moveItem<T>(list: T[], from: number, to: number): T[] {
  if (from === to || from < 0 || to < 0 || from >= list.length || to >= list.length) return list
  const next = list.slice()
  const [item] = next.splice(from, 1)
  next.splice(to, 0, item)
  return next
}

export function emptySetup(): AdventureSetup {
  return { title: '', genre: '', premise: '', character: '', tone: '', length: 'medium', choiceCount: 3, modelId: null, temperature: null }
}

/** The form after a starter idea was picked: the starter's fields over the rest. */
export function setupFromStarter(setup: AdventureSetup, s: AdventureStarter): AdventureSetup {
  return { ...setup, title: s.title, genre: s.genre, premise: s.premise, character: s.character, tone: s.tone }
}

/** Why the new-adventure form cannot be started yet; null when it can. */
export function setupProblem(setup: AdventureSetup, fromProject: boolean, folder: string | null): string | null {
  if (fromProject && !folder) return 'Pick a Writing project.'
  if (!setup.title.trim()) return 'Give the adventure a title.'
  if (!fromProject && !setup.premise.trim()) return 'Say what the story is about.'
  return null
}

/** The setup to send: trimmed text, choice count kept inside 2..4. */
export function cleanSetup(setup: AdventureSetup): AdventureSetup {
  return {
    ...setup,
    title: setup.title.trim(),
    genre: setup.genre.trim(),
    premise: setup.premise.trim(),
    character: setup.character.trim(),
    tone: setup.tone.trim(),
    choiceCount: Math.min(4, Math.max(2, Math.round(setup.choiceCount) || 3))
  }
}

/** The node a "Go back here" or sibling switch lands on: where the story stands after `nodeId` is picked as a path. */
export function siblingIndex(adv: Adventure, nodeId: string): { index: number; count: number; prev: string | null; next: string | null } {
  const n = adv.nodes[nodeId]
  if (!n || n.parentId === null) return { index: 0, count: 1, prev: null, next: null }
  const sibs = childrenOf(adv, n.parentId)
  const index = Math.max(0, sibs.findIndex((s) => s.id === nodeId))
  return { index, count: sibs.length, prev: sibs[index - 1]?.id ?? null, next: sibs[index + 1]?.id ?? null }
}

/** The state edited in the dialog, kept in a shape a form can hold (stat values as text). */
export interface StateDraft {
  inventory: string[]
  flags: string[]
  stats: { name: string; value: string }[]
}

export function draftOfState(st: AdventureState): StateDraft {
  return {
    inventory: st.inventory.slice(),
    flags: st.flags.slice(),
    stats: Object.entries(st.stats).map(([name, value]) => ({ name, value: String(value) }))
  }
}

/** The state a draft means: empty lines dropped, stat names unique (the last one wins), numbers read as numbers. */
export function stateOfDraft(d: StateDraft): AdventureState {
  const clean = (list: string[]): string[] => {
    const out: string[] = []
    for (const s of list.map((x) => x.trim())) if (s && !out.some((o) => o.toLowerCase() === s.toLowerCase())) out.push(s)
    return out
  }
  const stats: Record<string, StatValue> = {}
  for (const { name, value } of d.stats) {
    const n = name.trim()
    if (!n) continue
    const existing = Object.keys(stats).find((k) => k.toLowerCase() === n.toLowerCase())
    if (existing) delete stats[existing]
    stats[n] = parseStat(value)
  }
  return { inventory: clean(d.inventory), flags: clean(d.flags), stats }
}

export function sameState(a: AdventureState, b: AdventureState): boolean {
  return JSON.stringify([a.inventory, a.flags, Object.entries(a.stats)]) === JSON.stringify([b.inventory, b.flags, Object.entries(b.stats)])
}
