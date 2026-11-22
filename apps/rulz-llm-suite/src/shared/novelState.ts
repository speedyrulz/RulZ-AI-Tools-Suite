// Tracked state (state.json): pure reading-order logic shared by main and the renderer.
// Design: docs/writing-mode-design.md, "Stage 3 design", sections "Reading-order semantics" and "Tracked state in requests".
import type { BibleEntry, NovelOutline, StateFact, StateFactStatus } from './novel'

/** Reading-order index of every scene (chapter order, then scene order). A repeated scene id keeps its first position. */
export function scenePositions(outline: NovelOutline): Map<string, number> {
  const pos = new Map<string, number>()
  let i = 0
  for (const c of outline.chapters) {
    for (const s of c.scenes) {
      if (!pos.has(s.id)) pos.set(s.id, i)
      i++
    }
  }
  return pos
}

/**
 * 'detached': the entity, about, from or until names something that is not in the bible / outline.
 * 'inverted': until comes before from in reading order (after a scene move).
 * Only 'ok' facts are used in requests.
 */
export function factStatus(fact: StateFact, pos: Map<string, number>, entryIds: Set<string>): StateFactStatus {
  if (!entryIds.has(fact.entity)) return 'detached'
  if (fact.about != null && !entryIds.has(fact.about)) return 'detached'
  const from = pos.get(fact.from)
  if (from === undefined) return 'detached'
  if (fact.until != null) {
    const until = pos.get(fact.until)
    if (until === undefined) return 'detached'
    if (until < from) return 'inverted'
  }
  return 'ok'
}

/**
 * Location is single-valued: among the started facts, per entity only the location fact with the latest `from`
 * stays (a tie goes to the later fact in the list). It is applied before the `until` test, so an older open
 * location does not resurface when the latest one has ended.
 */
function singleLocation(facts: StateFact[], pos: Map<string, number>): StateFact[] {
  const best = new Map<string, StateFact>()
  for (const f of facts) {
    if (f.kind !== 'location') continue
    const cur = best.get(f.entity)
    if (!cur || (pos.get(f.from) ?? -1) >= (pos.get(cur.from) ?? -1)) best.set(f.entity, f)
  }
  return facts.filter((f) => f.kind !== 'location' || best.get(f.entity) === f)
}

function okFacts(facts: StateFact[], outline: NovelOutline, bible: BibleEntry[]): { ok: StateFact[]; pos: Map<string, number> } {
  const pos = scenePositions(outline)
  const ids = new Set(bible.map((b) => b.id))
  return { ok: facts.filter((f) => factStatus(f, pos, ids) === 'ok'), pos }
}

/** Facts holding before the scene: pos(from) < pos(S) and (until null or pos(until) >= pos(S)). Unknown scene: none. */
export function stateBefore(facts: StateFact[], sceneId: string, outline: NovelOutline, bible: BibleEntry[]): StateFact[] {
  const { ok, pos } = okFacts(facts, outline, bible)
  const at = pos.get(sceneId)
  if (at === undefined) return []
  const started = singleLocation(ok.filter((f) => (pos.get(f.from) as number) < at), pos)
  return started.filter((f) => f.until == null || (pos.get(f.until) as number) >= at)
}

/** Facts holding after the scene: pos(from) <= pos(S) and (until null or pos(until) > pos(S)). Unknown scene: none. */
export function stateAfter(facts: StateFact[], sceneId: string, outline: NovelOutline, bible: BibleEntry[]): StateFact[] {
  const { ok, pos } = okFacts(facts, outline, bible)
  const at = pos.get(sceneId)
  if (at === undefined) return []
  const started = singleLocation(ok.filter((f) => (pos.get(f.from) as number) <= at), pos)
  return started.filter((f) => f.until == null || (pos.get(f.until) as number) > at)
}

/** The facts recorded from, and ended at, this scene (so a re-run of Update state does not repeat them). */
export function factsAtScene(facts: StateFact[], sceneId: string): { added: StateFact[]; ended: StateFact[] } {
  return { added: facts.filter((f) => f.from === sceneId), ended: facts.filter((f) => f.until === sceneId) }
}

/** Lines for one entity's facts: "- location: the lighthouse", "- relationship (Oskar Lind): distrusts him". */
export function formatStateLines(facts: StateFact[], bible: BibleEntry[]): string {
  const names = new Map(bible.map((b) => [b.id, b.name]))
  return facts
    .map((f) => {
      if (f.kind === 'relationship' && f.about) return `- relationship (${names.get(f.about) ?? f.about}): ${f.value}`
      return `- ${f.kind}: ${f.value}`
    })
    .join('\n')
}
