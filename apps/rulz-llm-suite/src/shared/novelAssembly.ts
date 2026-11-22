// Writing mode: turns the counted pieces of a project into one chat request that fits the context.
// Pure and deterministic: the renderer counts every piece with the real tokenizer first, so there is
// no tokenizer and no retry loop here. Algorithm and fill order: docs/writing-mode-design.md.
import type {
  AssemblyDropped,
  AssemblyEntry,
  AssemblyInput,
  AssemblyPart,
  AssemblyPartKind,
  AssemblyResult,
  AssemblyStateGroup,
  AssemblySummary,
  Counted
} from './novel'

// The Markdown headings and the blank lines between items are text too. They are not counted by the
// renderer, so the assembler estimates them: ceil(chars / 3) of everything it adds. Three characters
// per token is on the high side for English, so the estimate errs towards a smaller request. The
// estimate is part of every part's `tokens`, of the caps and of `totalTokens`. The renderer still
// counts the whole request once more before sending.
const CHARS_PER_TOKEN = 3
const scaffoldTokens = (chars: number): number => Math.ceil(chars / CHARS_PER_TOKEN)

/** The "\n\n" in front of an item. */
const SEPARATOR = scaffoldTokens(2)

const HEADING = {
  bible: 'Story bible',
  story: 'Story so far',
  state: 'State before this scene',
  scene: 'This scene',
  relevant: 'Also relevant',
  prose: 'The text so far',
  wholeScene: 'The scene text',
  after: 'Text after the cursor',
  task: 'Your task'
} as const

/** "## Heading" plus the blank line after it. */
const headingTokens = (heading: string): number => scaffoldTokens(`## ${heading}`.length + 2)

/** Tokens of a section holding `items`: its heading and one separator per item. 0 when it has none. */
function sectionTokens(heading: string, items: Counted[]): number {
  if (items.length === 0) return 0
  return headingTokens(heading) + items.reduce((total, item) => total + item.tokens + SEPARATOR, 0)
}

const sumTokens = (items: Counted[]): number => items.reduce((total, item) => total + item.tokens, 0)

// ---------------------------------------------------------------------------------------------
// Prose anchor
// ---------------------------------------------------------------------------------------------

/**
 * Index of the first prose block to include. The start is chosen from anchor points that depend only
 * on the blocks before them (every block that holds a multiple of `anchorStep` tokens counted from
 * the first block; the step is capped at half the budget), and it is the first anchor point whose tail fits `tailBudget`. So appending blocks
 * never moves the start until the tail no longer fits; then it jumps to the next anchor point, about
 * one step further on. That keeps the beginning of the request the same across successive Continue
 * presses, so llama-server's prompt cache can reuse it. The tail ends up between budget - step and
 * budget tokens, except when a single block is bigger than the budget: the last block is kept anyway.
 */
export function anchorProse(blocks: Counted[], tailBudget: number, anchorStep: number): number {
  if (blocks.length === 0) return 0
  const budget = Math.max(0, tailBudget)
  // A step over half the budget would leave almost no anchor points that fit.
  const step = Math.min(Math.max(1, anchorStep), Math.max(1, Math.floor(budget / 2)))
  const total = sumTokens(blocks)
  if (total <= budget) return 0

  let before = 0
  for (let i = 0; i < blocks.length; i++) {
    // Block i is an anchor point if it holds a multiple of the step (so a big block that straddles
    // a boundary is a candidate too); block 0 always is.
    const holdsGridPoint = Math.ceil(before / step) * step < before + blocks[i].tokens
    if ((i === 0 || holdsGridPoint) && total - before <= budget) return i
    before += blocks[i].tokens
  }

  // No anchor point fits (the blocks near the end are big): fall back to the earliest block that
  // fits, or the last block alone. The fallback is the only case where the start is not on the grid.
  let tail = 0
  for (let i = blocks.length - 1; i >= 0; i--) {
    if (tail + blocks[i].tokens > budget) return Math.min(i + 1, blocks.length - 1)
    tail += blocks[i].tokens
  }
  return 0
}

// ---------------------------------------------------------------------------------------------
// Entries and summaries
// ---------------------------------------------------------------------------------------------

interface EntryChoice {
  entry: AssemblyEntry
  /** The full text or the short version. */
  piece: Counted
  cut: boolean
}

/** Actions that need the whole scene text in the request (stage 3). */
const WHOLE_SCENE_ACTIONS = ['checkScene', 'updateState']

/** The share of the available context the state of fixed entities may take (no setting, see the design). */
const STATE_SHARE = 0.08

const STATE_LABEL = 'State before this scene:'

// Fixed before triggered happens by group; inside a group: voice targets first, then priority high first,
// then name, then id. (Groups never move: a voice target only goes first inside its own group.)
function byPriority(a: AssemblyEntry, b: AssemblyEntry): number {
  if (Boolean(a.voice) !== Boolean(b.voice)) return a.voice ? -1 : 1
  if (a.priority !== b.priority) return b.priority - a.priority
  return a.name.localeCompare(b.name, 'en') || a.id.localeCompare(b.id, 'en')
}

/**
 * Fits one group of entries into `cap` tokens (heading included). Lowest priority first, entries
 * are cut to their first paragraph; if that is still too much, lowest priority first, they are dropped.
 */
function fitEntries(
  group: AssemblyEntry[],
  heading: string,
  cap: number,
  kind: AssemblyPartKind,
  dropped: AssemblyDropped[]
): EntryChoice[] {
  const items: EntryChoice[] = group.map((entry) => ({ entry, piece: entry, cut: false }))
  const cost = (): number => sectionTokens(heading, items.map((item) => item.piece))

  for (let i = items.length - 1; i >= 0 && cost() > cap; i--) {
    const item = items[i]
    if (item.entry.short.tokens < item.piece.tokens) {
      item.piece = item.entry.short
      item.cut = true
    }
  }
  while (items.length > 0 && cost() > cap) {
    const item = items.pop() as EntryChoice
    dropped.push({ kind, label: item.entry.name, how: 'dropped', reason: 'budget' })
  }
  // Drops free room: bring back full text, highest priority first, while it fits.
  for (const item of items) {
    if (!item.cut) continue
    item.piece = item.entry
    if (cost() > cap) item.piece = item.entry.short
    else item.cut = false
  }
  for (const item of items) {
    if (item.cut) dropped.push({ kind, label: item.entry.name, how: 'cut', reason: 'budget' })
  }
  return items
}

/** Drops the oldest summaries until the section fits `room`. */
function fitSummaries(summaries: AssemblySummary[], room: number, dropped: AssemblyDropped[]): AssemblySummary[] {
  const kept = [...summaries]
  while (kept.length > 0 && sectionTokens(HEADING.story, kept) > room) {
    const oldest = kept.shift() as AssemblySummary
    dropped.push({ kind: 'storySoFar', label: oldest.label, how: 'dropped', reason: 'budget' })
  }
  return kept
}

// ---------------------------------------------------------------------------------------------
// Tracked state (stage 3)
// ---------------------------------------------------------------------------------------------

/** A triggered entry with its state lines appended; the short form keeps them too (they go with the entry). */
function withState(entry: AssemblyEntry, group: AssemblyStateGroup | undefined): AssemblyEntry {
  if (!group || !group.text.trim()) return entry
  const extra = `\n\n${STATE_LABEL}\n${group.text}`
  const cost = group.tokens + scaffoldTokens(extra.length - group.text.length)
  return {
    ...entry,
    text: entry.text + extra,
    tokens: entry.tokens + cost,
    short: { text: entry.short.text + extra, tokens: entry.short.tokens + cost }
  }
}

interface StateItem extends Counted {
  group: AssemblyStateGroup
  /** 0 = POV, 1 = present, 2 = anything else; then entry priority. */
  rank: number
  priority: number
}

function stateRank(entry: AssemblyEntry, povId?: string | null, presentIds?: string[]): number {
  if (povId !== undefined || presentIds !== undefined) {
    if (povId === entry.id) return 0
    if (presentIds?.includes(entry.id)) return 1
    return 2
  }
  if (entry.reason === 'pov') return 0
  if (entry.reason === 'present') return 1
  return 2
}

/** "### Name" then the lines; a group text that already starts with a heading is used as it is. */
function stateItem(
  group: AssemblyStateGroup,
  entry: AssemblyEntry,
  povId?: string | null,
  presentIds?: string[]
): StateItem {
  const text = group.text.trimStart().startsWith('###') ? group.text : `### ${group.name}\n${group.text}`
  const tokens = group.tokens + (text === group.text ? 0 : scaffoldTokens(text.length - group.text.length))
  return { text, tokens, group, rank: stateRank(entry, povId, presentIds), priority: entry.priority }
}

/**
 * Fits the state of the fixed entities into `cap`: whole groups are dropped, POV first kept, then present,
 * then by entry priority. Returns the groups that stay, in that order.
 */
function fitState(items: StateItem[], cap: number, dropped: AssemblyDropped[]): StateItem[] {
  const kept = [...items].sort(
    (a, b) =>
      a.rank - b.rank ||
      b.priority - a.priority ||
      a.group.name.localeCompare(b.group.name, 'en') ||
      a.group.entity.localeCompare(b.group.entity, 'en')
  )
  while (kept.length > 0 && sectionTokens(HEADING.state, kept) > cap) {
    const last = kept.pop() as StateItem
    dropped.push({ kind: 'state', label: last.group.name, how: 'dropped', reason: 'budget' })
  }
  return kept
}

// ---------------------------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------------------------

/**
 * The input plus the scene POV and present entry ids, so state is ranked by who is in the scene even when an
 * entry reason says 'always' (an always entry that is also the POV is still the POV).
 */
export type AssemblyRequest = AssemblyInput & { povId?: string | null; presentIds?: string[] }

export function assembleRequest(input: AssemblyRequest): AssemblyResult {
  const { system, task, scene, summaries, proseBlocks, proseAfter, budget } = input
  const available =
    input.contextLength -
    input.templateOverhead -
    budget.replyTokens -
    Math.max(256, Math.ceil(input.contextLength * 0.03))
  const warnings: string[] = []
  const dropped: AssemblyDropped[] = []

  // 1 + 2. System, task and scene take what they need. Only a warning when they are large.
  const sceneCost = scene.text ? sectionTokens(HEADING.scene, [scene]) : 0
  const taskCost = task.text ? sectionTokens(HEADING.task, [task]) : 0
  const base = system.tokens + sceneCost + taskCost
  if (system.tokens > available * 0.3) {
    warnings.push('The system text and style notes use more than 30% of the available context.')
  }
  if (task.tokens + scene.tokens > available * 0.15) {
    warnings.push('The task and the scene use more than 15% of the available context.')
  }

  // 3. Prose tail: its cap is a share of what is left, at least 800 tokens (or all the prose).
  const afterCost = proseAfter.text ? sectionTokens(HEADING.after, [proseAfter]) : 0
  // Each block carries the separator in front of it, so the anchor sees the real cost of a block.
  const blocks: Counted[] = proseBlocks.map((block) => ({ text: block.text, tokens: block.tokens + SEPARATOR }))
  const wholeScene = WHOLE_SCENE_ACTIONS.includes(input.action)
  const proseHeading = wholeScene ? HEADING.wholeScene : HEADING.prose
  const blocksHeading = blocks.length > 0 ? headingTokens(proseHeading) : 0
  const proseTotal = blocksHeading + sumTokens(blocks) + afterCost
  const floor = Math.min(800, proseTotal)
  if (base + floor > available) {
    warnings.push('The system text, task, scene and the minimum of prose do not fit in the available context.')
  }
  const proseCap = Math.max(floor, Math.floor(Math.max(0, available - base) * budget.proseShare))
  // The room reserved for the tail while the other parts are fitted: the whole cap whenever there is
  // any prose, not the tail that happens to be chosen, so entries and summaries do not change while the
  // prose grows or steps. What the tail does not use flows back to it at the end.
  // A whole-scene action (check, update state) needs all of the scene text, so all of it is reserved.
  const reserve = proseTotal > 0 ? (wholeScene ? Math.min(proseTotal, Math.max(0, available - base)) : proseCap) : 0
  let remaining = Math.max(0, available - base - reserve)

  // 4 + 5. Entries share bibleShare of the available context: fixed first, triggered get the rest.
  const bibleCap = Math.max(0, Math.floor(available * budget.bibleShare))
  const fixed = input.entries.filter((entry) => entry.fixed).sort(byPriority)
  const groupOf = new Map<string, AssemblyStateGroup>()
  for (const group of input.stateGroups ?? []) if (!groupOf.has(group.entity)) groupOf.set(group.entity, group)
  const triggered = input.entries
    .filter((entry) => !entry.fixed)
    .map((entry) => withState(entry, groupOf.get(entry.id)))
    .sort(byPriority)
  const fixedCap = Math.max(0, Math.min(bibleCap, remaining))
  const fixedChoices = fitEntries(fixed, HEADING.bible, fixedCap, 'fixedEntries', dropped)
  const fixedCost = sectionTokens(HEADING.bible, fixedChoices.map((item) => item.piece))
  remaining -= fixedCost

  // Tracked state of the fixed entries: its own cap, fitted before the triggered entries and the story so
  // far, charged at its actual cost. A group goes when its entry did.
  const keptFixed = new Map(fixedChoices.map((item) => [item.entry.id, item.entry]))
  const stateItems: StateItem[] = []
  for (const group of input.stateGroups ?? []) {
    if (!group.text.trim()) continue
    if (!fixed.some((entry) => entry.id === group.entity)) continue
    const owner = keptFixed.get(group.entity)
    if (!owner) {
      dropped.push({ kind: 'state', label: group.name, how: 'dropped', reason: 'its entry was dropped' })
      continue
    }
    stateItems.push(stateItem(group, owner, input.povId, input.presentIds))
  }
  const stateCap = Math.max(0, Math.min(Math.floor(available * STATE_SHARE), remaining))
  const stateKept = fitState(stateItems, stateCap, dropped)
  const stateCost = sectionTokens(HEADING.state, stateKept)
  remaining -= stateCost

  const triggeredCap = Math.max(0, Math.min(bibleCap - fixedCost, remaining))
  const triggeredChoices = fitEntries(triggered, HEADING.relevant, triggeredCap, 'triggeredEntries', dropped)
  const triggeredCost = sectionTokens(HEADING.relevant, triggeredChoices.map((item) => item.piece))
  remaining -= triggeredCost

  // 6. Story so far takes everything left, oldest summaries dropped first.
  const storyRoom = Math.max(0, remaining)
  const storyKept = fitSummaries(summaries, storyRoom, dropped)
  const storyCost = sectionTokens(HEADING.story, storyKept)

  // Whatever is still unused goes back to the prose tail: the anchor is chosen again with the larger
  // budget. It does not depend on which tail was reserved above, only on what the other parts took.
  const proseRoom = Math.max(floor, available - base - fixedCost - stateCost - triggeredCost - storyCost)
  const tailBudget = Math.max(0, proseRoom - afterCost - blocksHeading)
  const proseStart = anchorProse(blocks, tailBudget, input.anchorStep)
  const proseBlockList = proseBlocks.slice(proseStart)
  const proseCost =
    (proseBlockList.length > 0 ? headingTokens(proseHeading) : 0) +
    blocks.slice(proseStart).reduce((total, block) => total + block.tokens, 0) +
    afterCost
  if (wholeScene && (proseStart > 0 || proseCost > proseRoom)) {
    warnings.push('The whole scene does not fit in the available context; this action needs all of it.')
  }
  if (proseStart > 0) {
    dropped.push({
      kind: 'prose',
      label: `Earlier text (${proseStart} ${proseStart === 1 ? 'paragraph' : 'paragraphs'})`,
      how: 'cut',
      reason: 'budget'
    })
  }

  // Parts, in fill order, for the budget bar. `tokens` includes each part's own headings/separators.
  const parts: AssemblyPart[] = []
  const addPart = (kind: AssemblyPartKind, tokens: number, cap: number | null, present: boolean): void => {
    if (present) parts.push({ kind, tokens, cap })
  }
  addPart('system', system.tokens, null, Boolean(system.text))
  addPart('task', taskCost, null, Boolean(task.text))
  addPart('scene', sceneCost, null, Boolean(scene.text))
  addPart('prose', proseCost, proseRoom, proseCost > 0)
  addPart('fixedEntries', fixedCost, fixedCap, fixedChoices.length > 0)
  addPart('state', stateCost, stateCap, stateKept.length > 0)
  addPart('triggeredEntries', triggeredCost, triggeredCap, triggeredChoices.length > 0)
  addPart('storySoFar', storyCost, storyRoom, storyKept.length > 0)

  // Messages: the most stable text first, so the prompt cache reuses the longest prefix.
  const sections: string[] = []
  const addSection = (heading: string, texts: string[]): void => {
    if (texts.length > 0) sections.push(`## ${heading}\n\n${texts.join('\n\n')}`)
  }
  addSection(HEADING.bible, fixedChoices.map((item) => item.piece.text))
  addSection(HEADING.story, storyKept.map((item) => item.text))
  addSection(HEADING.state, stateKept.map((item) => item.text))
  addSection(HEADING.scene, scene.text ? [scene.text] : [])
  addSection(HEADING.relevant, triggeredChoices.map((item) => item.piece.text))
  addSection(proseHeading, proseBlockList.map((block) => block.text))
  addSection(HEADING.after, proseAfter.text ? [proseAfter.text] : [])
  addSection(HEADING.task, task.text ? [task.text] : [])

  // `available` already has the template overhead taken out, so the content is what must stay below
  // it; `totalTokens` adds the overhead back to show the whole request. The UI compares
  // totalTokens - templateOverhead with `available`.
  const content = parts.reduce((total, part) => total + part.tokens, 0)
  const totalTokens = content + input.templateOverhead
  if (content > available && !warnings.some((warning) => warning.includes('do not fit'))) {
    warnings.push('The request is larger than the available context.')
  }

  return {
    messages: [
      { role: 'system', content: system.text },
      { role: 'user', content: sections.join('\n\n') }
    ],
    parts,
    dropped,
    includedEntries: [...fixedChoices, ...triggeredChoices].map((item) => item.entry.id),
    proseStart,
    totalTokens,
    available,
    warnings
  }
}
