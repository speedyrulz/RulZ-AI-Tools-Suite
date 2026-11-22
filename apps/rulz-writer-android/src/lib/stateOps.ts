import type { BibleEntry, NovelProjectState, StateFact, StateKind, StateOp } from '@shared/novel'
import { factsAtScene, stateAfter, stateBefore } from '@shared/novelState'

// Tracked state (state.json) for the scene being written, and the manual edits the phone offers.

export const STATE_KINDS: { value: StateKind; label: string }[] = [
  { value: 'location', label: 'Location' },
  { value: 'knows', label: 'Knows' },
  { value: 'carries', label: 'Carries' },
  { value: 'condition', label: 'Condition' },
  { value: 'relationship', label: 'Relationship' },
  { value: 'other', label: 'Other' }
]

export function kindLabel(kind: StateKind): string {
  return STATE_KINDS.find((k) => k.value === kind)?.label ?? kind
}

export interface SceneFacts {
  /** True at the start of the scene. */
  before: StateFact[]
  /** Recorded from this scene on. */
  added: StateFact[]
  /** Ended at this scene. */
  ended: StateFact[]
  /** True once the scene is over. */
  after: StateFact[]
}

/** What the tracked state says around one scene. Facts that don't validate are left out by the shared helpers. */
export function factsForScene(state: NovelProjectState, sceneId: string): SceneFacts {
  const facts = state.state.facts
  const { added, ended } = factsAtScene(facts, sceneId)
  return {
    before: stateBefore(facts, sceneId, state.outline, state.bible),
    added,
    ended,
    after: stateAfter(facts, sceneId, state.outline, state.bible)
  }
}

export function entryName(bible: BibleEntry[], id: string | null): string {
  if (!id) return ''
  return bible.find((e) => e.id === id)?.name ?? id
}

/** "Mara Venn: location — the lighthouse", "Mara Venn: relationship with Oskar Lind — distrusts him". */
export function factLine(fact: StateFact, bible: BibleEntry[]): string {
  const who = entryName(bible, fact.entity)
  const kind = fact.kind === 'relationship' && fact.about ? `relationship with ${entryName(bible, fact.about)}` : fact.kind
  return `${who}: ${kind} — ${fact.value}`
}

/** The entry a relationship is with: the chosen one unless it is the entry the fact is about, else the first other entry. */
export function otherEntry(bible: BibleEntry[], entityId: string, chosen: string): string {
  const others = bible.filter((b) => b.id !== entityId)
  return others.some((b) => b.id === chosen) ? chosen : (others[0]?.id ?? '')
}

export function newFactId(): string {
  const bytes = new Uint8Array(8)
  globalThis.crypto.getRandomValues(bytes)
  let out = ''
  for (const b of bytes) out += '0123456789abcdefghijklmnopqrstuvwxyz'[b % 36]
  return `f${out}`
}

export interface NewFact {
  entity: string
  kind: StateKind
  value: string
  about: string | null
}

/** The value is one line of 1-300 characters; relationships need the other entry. Returns a problem or null. */
export function validateFact(f: NewFact): string | null {
  const value = f.value.replace(/\s+/g, ' ').trim()
  if (!f.entity) return 'Choose who or what this is about.'
  if (!value) return 'Write the fact.'
  if (value.length > 300) return 'Keep it under 300 characters.'
  if (f.kind === 'relationship' && !f.about) return 'Choose the other character for a relationship.'
  return null
}

export function addFactOps(f: NewFact, sceneId: string): StateOp[] {
  return [
    {
      op: 'add',
      fact: {
        id: newFactId(),
        entity: f.entity,
        kind: f.kind,
        value: f.value.replace(/\s+/g, ' ').trim(),
        about: f.kind === 'relationship' ? f.about : null,
        from: sceneId,
        until: null
      }
    }
  ]
}

export function endFactOps(factId: string, sceneId: string): StateOp[] {
  return [{ op: 'end', factId, until: sceneId }]
}

export function removeFactOps(factId: string): StateOp[] {
  return [{ op: 'remove', factId }]
}
