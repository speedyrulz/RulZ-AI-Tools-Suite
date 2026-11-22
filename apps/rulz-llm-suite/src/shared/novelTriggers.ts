// Bible retrieval by keyword (no embeddings). Pure; the renderer calls it with the scan text.
import type { BibleEntry, NovelAction, NovelScene } from './novel'

export interface TriggerCandidate {
  entry: BibleEntry
  reason: string
  fixed: boolean
  priority: number
}

const regexCache = new Map<string, RegExp | null>()

// Whole-word, case-insensitive, Unicode-aware. \b is ASCII-only, so the boundaries are lookarounds
// on letters and digits. Spaces in a phrase match any run of whitespace ("the Salt Witch" also
// matches across a line break).
function phraseRegex(phrase: string): RegExp | null {
  const cached = regexCache.get(phrase)
  if (cached !== undefined) return cached
  const words = phrase.trim().split(/\s+/).filter(Boolean)
  let regex: RegExp | null = null
  if (words.length > 0) {
    const body = words.map((word) => word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('\\s+')
    regex = new RegExp(`(?<![\\p{L}\\p{N}])${body}(?![\\p{L}\\p{N}])`, 'iu')
  }
  if (regexCache.size > 500) regexCache.clear()
  regexCache.set(phrase, regex)
  return regex
}

function mentions(text: string, phrase: string): boolean {
  const regex = phraseRegex(phrase)
  return regex !== null && regex.test(text)
}

// Why an entry is in every request for this scene, or '' if it is not fixed. Sticky entries are
// not fixed: the design lists them with the triggered entries ("Also relevant"), where they stay put.
function fixedReason(entry: BibleEntry, scene: NovelScene | null): string {
  if (entry.always) return 'always'
  if (!scene) return ''
  if (scene.pov === entry.id) return 'pov'
  if (scene.present.includes(entry.id)) return 'present'
  return ''
}

function byOrder(a: TriggerCandidate, b: TriggerCandidate): number {
  if (a.fixed !== b.fixed) return a.fixed ? -1 : 1
  if (a.priority !== b.priority) return b.priority - a.priority
  return a.entry.name.localeCompare(b.entry.name, 'en') || a.entry.id.localeCompare(b.entry.id, 'en')
}

export function matchEntries(
  entries: BibleEntry[],
  scanText: string,
  scene: NovelScene | null
): TriggerCandidate[] {
  const found = new Map<string, TriggerCandidate>()

  // Pass 1: fixed entries, then sticky ones, then keyword matches. A fixed entry is never re-added as triggered.
  for (const entry of entries) {
    if (found.has(entry.id)) continue
    const reason = fixedReason(entry, scene)
    if (reason) {
      found.set(entry.id, { entry, reason, fixed: true, priority: entry.priority })
      continue
    }
    if (scene?.sticky.includes(entry.id)) {
      found.set(entry.id, { entry, reason: 'sticky', fixed: false, priority: entry.priority })
      continue
    }
    const word = [entry.name, ...entry.aliases, ...entry.triggers].find((value) => mentions(scanText, value))
    if (word !== undefined) {
      found.set(entry.id, { entry, reason: `trigger: ${word.trim()}`, fixed: false, priority: entry.priority })
    }
  }

  // Pass 2, depth 1: names and aliases in the bodies of the keyword-matched and sticky entries pull in more
  // entries at half priority. Fixed entries are not scanned, and links are not followed further.
  const sources = [...found.values()].filter((candidate) => !candidate.fixed)
  for (const source of sources) {
    for (const entry of entries) {
      if (found.has(entry.id)) continue
      if (![entry.name, ...entry.aliases].some((value) => mentions(source.entry.body, value))) continue
      found.set(entry.id, {
        entry,
        reason: `via: ${source.entry.name}`,
        fixed: false,
        priority: Math.floor(entry.priority / 2)
      })
    }
  }

  return [...found.values()].sort(byOrder)
}

/**
 * Voice targets (stage 3), character entries only: for Dialogue the scene's POV and present characters; for
 * any other action with a selection, the characters whose name or alias appears in it (whole word, any case).
 * Returns entry ids in bible order.
 */
export function voiceTargets(
  action: NovelAction,
  scene: NovelScene | null,
  selection: string,
  bible: BibleEntry[]
): string[] {
  const characters = bible.filter((entry) => entry.kind === 'character')
  if (action === 'dialogue') {
    if (!scene) return []
    const wanted = new Set<string>([...(scene.pov ? [scene.pov] : []), ...scene.present])
    return characters.filter((entry) => wanted.has(entry.id)).map((entry) => entry.id)
  }
  if (!selection.trim()) return []
  return characters
    .filter((entry) => [entry.name, ...entry.aliases].some((value) => mentions(selection, value)))
    .map((entry) => entry.id)
}
