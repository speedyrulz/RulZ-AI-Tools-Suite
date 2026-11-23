// Adventure mode, the pure part: what the model is told, how its replies are read, how the story is fitted to the
// context, the Markdown export and the starter ideas. No node imports and no model calls (the counting function
// comes in as an argument), so tests run it as plain functions. Design: docs/adventure-mode-design.md.
import {
  MAX_CHOICES,
  MIN_CHOICES,
  PASSAGE_WORDS,
  pathTo,
  type Adventure,
  type AdventureNode,
  type AdventureSetup,
  type AdventureStarter,
  type AdventureState,
  type NodeInput,
  type StateChanges,
  type StatValue
} from './adventure'
import { cleanProse } from './novelPrompts'

export interface StoryMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

// ---------------------------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------------------------

export const BEGIN_MESSAGE = 'Begin the story.'
export const LEFT_OUT_LINE = '(Earlier passages left out.)'

/** Fixed for the whole adventure (no state in it), so the model's prompt cache keeps the start of the prompt. */
export function systemMessage(setup: AdventureSetup): string {
  const words = PASSAGE_WORDS[setup.length] ?? PASSAGE_WORDS.medium
  const lines: string[] = ['You are the narrator of an interactive story that is played one passage at a time.', '']
  lines.push(`Genre: ${setup.genre.trim() || 'any'}`)
  if (setup.tone.trim()) lines.push(`Tone: ${setup.tone.trim()}`)
  if (setup.premise.trim()) lines.push(`Premise: ${setup.premise.trim()}`)
  lines.push(
    setup.character.trim()
      ? `The reader plays: ${setup.character.trim()}`
      : 'The reader\'s character is not described: invent a fitting one in the opening and keep to it.'
  )
  if (setup.world?.text.trim()) {
    lines.push('', 'The author\'s notes on the world follow. Use these characters and places; stay consistent with them.', '', setup.world.text.trim())
  }
  lines.push(
    '',
    'How to narrate:',
    '- Write in the second person ("you") and the present tense.',
    `- Each passage is about ${words} words.`,
    '- End every passage at a moment where the reader has to decide what to do next.',
    '- Never decide for the reader, never list the choices in the passage and never speak to the reader outside the story.',
    '- Write only the story text: no title, no label, no notes.',
    '- Stay consistent with everything that has already happened and with the tracked state the reader\'s messages give.'
  )
  return lines.join('\n')
}

/** The reader's step, as the user message that follows the passage it was taken from. */
export function stepLine(input: NodeInput): string {
  if (input.kind === 'start') return BEGIN_MESSAGE
  if (input.kind === 'choice') return `The reader chooses: ${input.text.trim()}`
  return `The reader does: ${input.text.trim()}`
}

const statText = (v: StatValue): string => String(v)

/** The tracked state as lines of text; `none` marks an empty list so the model does not invent items. */
export function stateLines(state: AdventureState): string[] {
  const stats = Object.entries(state.stats).map(([k, v]) => `${k} ${statText(v)}`)
  return [
    `Inventory: ${state.inventory.length ? state.inventory.join('; ') : 'nothing'}`,
    `Stats: ${stats.length ? stats.join('; ') : 'none'}`,
    `Facts: ${state.flags.length ? state.flags.join('; ') : 'none'}`
  ]
}

/** The last user message of a passage request: the reader's step, the state it must stay consistent with, the ask. */
export function passageFinal(input: NodeInput, state: AdventureState): string {
  if (input.kind === 'start') return BEGIN_MESSAGE
  return [
    stepLine(input),
    '',
    'Current state of the story. The story must stay consistent with this. The character only has these items.',
    ...stateLines(state),
    '',
    'Write the next passage.'
  ].join('\n')
}

/** The last user message of a choosing request (the passage just written is the last assistant message). */
export function choosingFinal(setup: AdventureSetup, stateBefore: AdventureState, opening: boolean): string {
  const n = Math.min(MAX_CHOICES, Math.max(MIN_CHOICES, Math.round(setup.choiceCount) || MIN_CHOICES))
  const lines: string[] = []
  if (!opening) {
    lines.push('State before that passage:', ...stateLines(stateBefore), '')
  }
  lines.push(
    'Now describe what comes after the passage you just wrote, as one JSON object and nothing else:',
    `- "choices": exactly ${n} options for what the reader can do next. Each is short (under about 15 words), starts with a verb and differs clearly from the others. If the story ends in this passage, give an empty list.`,
    '- "ended": true only when the story reaches a real end (a death, a victory, a natural close); otherwise false.',
    '- "changes": what the passage changed in the tracked state. Change only what the passage actually changed. addItems and removeItems list item names; setStats is a list of {name, value} (keep numbers as numbers where they are numbers) and removeStats lists stat names; addFlags and removeFlags list facts that became true or stopped being true.'
  )
  if (opening) lines.push('This is the opening: set a few starting stats and items that fit the story (for example Health, and what the character carries).')
  return lines.join('\n')
}

/** The JSON schema of the choosing reply. setStats is a list, because constrained output handles fixed keys far better. */
export function choosingSchema(choiceCount: number): object {
  const strings = { type: 'array', items: { type: 'string' } }
  const n = Math.min(MAX_CHOICES, Math.max(MIN_CHOICES, Math.round(choiceCount) || MIN_CHOICES))
  return {
    type: 'object',
    properties: {
      // minItems too: without it the grammar allows an empty list, which reads as "no choices".
      choices: { type: 'array', items: { type: 'string', minLength: 1 }, minItems: MIN_CHOICES, maxItems: n },
      ended: { type: 'boolean' },
      changes: {
        type: 'object',
        properties: {
          addItems: strings,
          removeItems: strings,
          setStats: {
            type: 'array',
            items: {
              type: 'object',
              properties: { name: { type: 'string' }, value: { anyOf: [{ type: 'number' }, { type: 'string' }] } },
              required: ['name', 'value']
            }
          },
          removeStats: strings,
          addFlags: strings,
          removeFlags: strings
        },
        required: ['addItems', 'removeItems', 'setStats', 'removeStats', 'addFlags', 'removeFlags']
      }
    },
    required: ['choices', 'ended', 'changes']
  }
}

// ---------------------------------------------------------------------------------------------
// Fitting the story to the context
// ---------------------------------------------------------------------------------------------

/** Passages are dropped in blocks of this many, so the cut point (and with it the cached prompt) moves only now and then. */
export const DROP_BLOCK = 8

const estimateMessages = (messages: StoryMessage[]): number => messages.reduce((n, m) => n + Math.ceil(m.content.length / 4) + 4, 0)

export interface StoryFit {
  messages: StoryMessage[]
  /** Passages left out of the middle (or the start) of the story. */
  dropped: number
  /** Not even the system message, the newest passage and the last message fit. */
  tooBig: boolean
  /** The model could not count, so the sizes are estimates (4 characters a token). */
  estimated: boolean
}

/**
 * The messages of one request: the system message, the story so far as alternating turns, and `final` as the last
 * user message. `path` is the passages already written, root first (the opening is `path[0]`; its user turn is
 * "Begin the story."). The newest passages stay; when the whole story does not fit, the oldest go in blocks of eight,
 * keeping the opening while it fits, and one line takes their place.
 */
export async function fitStory(opts: {
  system: string
  path: AdventureNode[]
  final: string
  countMessages: (messages: StoryMessage[]) => Promise<number | null>
  /** Most tokens the messages may take (context minus reply budget minus a margin). */
  limit: number
  /** The newest passage is never dropped (the choosing request analyzes it); when it cannot fit the result is tooBig. */
  keepNewest?: boolean
}): Promise<StoryFit> {
  const { system, path, final, limit } = opts
  const n = path.length

  const build = (keepOpening: boolean, from: number): { messages: StoryMessage[]; dropped: number } => {
    const kept: number[] = []
    if (keepOpening && n > 0) kept.push(0)
    for (let i = Math.max(from, keepOpening ? 1 : 0); i < n; i++) kept.push(i)
    const dropped = n - kept.length
    const messages: StoryMessage[] = [{ role: 'system', content: system }]
    // The gap sits before the first kept passage that comes after the dropped ones.
    const gapBefore = dropped > 0 ? Math.max(from, keepOpening ? 1 : 0) : -1
    for (const i of kept) {
      const step = i === 0 ? BEGIN_MESSAGE : stepLine(path[i].input)
      messages.push({ role: 'user', content: i === gapBefore ? `${LEFT_OUT_LINE}\n${step}` : step })
      messages.push({ role: 'assistant', content: path[i].text })
    }
    messages.push({ role: 'user', content: gapBefore === n ? `${LEFT_OUT_LINE}\n${final}` : final })
    return { messages, dropped }
  }

  // Candidates from "everything" to "almost nothing": the opening plus the newest passages, dropping 8, 16, ... from
  // right after the opening; then without the opening, still in blocks of 8 from the oldest; then only the newest
  // passage; then (unless keepNewest) none.
  const candidates: { keepOpening: boolean; from: number }[] = []
  for (let d = 0; ; d += DROP_BLOCK) {
    const from = 1 + d
    if (d > 0 && from > n - 1) break
    candidates.push({ keepOpening: true, from })
  }
  for (let from = DROP_BLOCK; from < n - 1; from += DROP_BLOCK) candidates.push({ keepOpening: false, from })
  candidates.push({ keepOpening: false, from: Math.max(0, n - 1) })
  if (!opts.keepNewest) candidates.push({ keepOpening: false, from: n })

  let estimated = false
  let last = build(true, 1)
  for (const c of candidates) {
    last = build(c.keepOpening, c.from)
    let tokens: number | null = null
    try {
      tokens = await opts.countMessages(last.messages)
    } catch {
      tokens = null
    }
    if (tokens === null) {
      estimated = true
      tokens = estimateMessages(last.messages)
    }
    if (tokens <= limit) return { messages: last.messages, dropped: last.dropped, tooBig: false, estimated }
  }
  return { messages: last.messages, dropped: last.dropped, tooBig: true, estimated }
}

// ---------------------------------------------------------------------------------------------
// Reading the model's replies
// ---------------------------------------------------------------------------------------------

export interface ChoosingReply {
  choices: string[]
  ended: boolean
  changes: StateChanges
}

export type ChoosingParse = { ok: true; value: ChoosingReply; warnings: string[] } | { ok: false; error: string }

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)

/** The text of the `{...}` that starts at `start`, honouring strings; null when it never closes. */
function balancedAt(text: string, start: number): string | null {
  let depth = 0
  let inString = false
  for (let i = start; i < text.length; i++) {
    const ch = text[i]
    if (inString) {
      if (ch === '\\') i++
      else if (ch === '"') inString = false
    } else if (ch === '"') inString = true
    else if (ch === '{') depth++
    else if (ch === '}' && --depth === 0) return text.slice(start, i + 1)
  }
  return null
}

function tryParse(text: string): unknown {
  try {
    return JSON.parse(text)
  } catch {
    return undefined
  }
}

/** The reply's JSON: the whole text, a fenced block, or the first `{...}` inside other text that looks like the answer. */
function findJson(raw: string): unknown {
  const text = raw.trim()
  const whole = tryParse(text)
  if (isObj(whole) || Array.isArray(whole)) return whole
  const fenced = /```(?:json)?\s*([\s\S]*?)```/i.exec(text)
  if (fenced) {
    const v = tryParse(fenced[1].trim())
    if (isObj(v) || Array.isArray(v)) return v
  }
  let firstObject: unknown
  for (let i = text.indexOf('{'); i !== -1; i = text.indexOf('{', i + 1)) {
    const piece = balancedAt(text, i)
    if (!piece) continue
    const v = tryParse(piece)
    if (!isObj(v)) continue
    if ('choices' in v || 'ended' in v || 'changes' in v) return v
    firstObject ??= v
  }
  return firstObject
}

const LIST_MARKER = /^\s*(?:[-*•]+|\d+[.):]|[A-Da-d][.)])\s+/

function cleanChoice(s: string): string {
  let t = s.replace(LIST_MARKER, '').replace(/\s+/g, ' ').trim()
  t = t.replace(/^[*_"'“‘]+/, '').replace(/[*_"'”’]+$/, '').trim()
  return t.length > 300 ? `${t.slice(0, 299).trimEnd()}…` : t
}

function cleanChoices(list: unknown[]): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const item of list) {
    const text = typeof item === 'string' ? item : isObj(item) && typeof item.text === 'string' ? item.text : ''
    const c = cleanChoice(text)
    if (!c || seen.has(c.toLowerCase())) continue
    seen.add(c.toLowerCase())
    out.push(c)
  }
  return out
}

const strings = (v: unknown): string[] => {
  const list = Array.isArray(v) ? v : typeof v === 'string' ? [v] : []
  return list.filter((x): x is string => typeof x === 'string').map((x) => x.trim()).filter(Boolean)
}

function statValue(v: unknown): StatValue | undefined {
  if (typeof v === 'number' && Number.isFinite(v)) return v
  if (typeof v === 'string') {
    const t = v.trim()
    if (!t) return undefined
    // A model that writes "12" for a number stat still means the number.
    return /^-?\d+(\.\d+)?$/.test(t) ? Number(t) : t
  }
  return undefined
}

function setStats(v: unknown): { name: string; value: StatValue }[] {
  const entries: [unknown, unknown][] = Array.isArray(v)
    ? v.filter(isObj).map((o) => [o.name, o.value] as [unknown, unknown])
    : isObj(v)
      ? Object.entries(v)
      : []
  const out: { name: string; value: StatValue }[] = []
  for (const [name, value] of entries) {
    const val = statValue(value)
    if (typeof name !== 'string' || !name.trim() || val === undefined) continue
    out.push({ name: name.trim(), value: val })
  }
  return out
}

/** Reads the changes object the way the model may have written it; missing fields become empty ones. */
export function normalizeChanges(v: unknown): StateChanges {
  const o = isObj(v) ? v : {}
  return {
    addItems: strings(o.addItems),
    removeItems: strings(o.removeItems),
    setStats: setStats(o.setStats),
    removeStats: strings(o.removeStats),
    addFlags: strings(o.addFlags),
    removeFlags: strings(o.removeFlags)
  }
}

/**
 * Reads the choosing reply. Forgiving: the object may sit in a fence or between sentences, choices may be objects or
 * numbered lines, and missing `changes` fields become empty ones. A reply with no choices that does not end the
 * story is refused (the caller leaves the passage as it is and says so).
 */
export function parseChoosing(raw: string, choiceCount: number): ChoosingParse {
  const warnings: string[] = []
  const found = findJson(raw)
  let obj: Record<string, unknown> | null = null
  if (isObj(found)) obj = found
  else if (Array.isArray(found)) obj = { choices: found }
  if (!obj) {
    // No JSON at all: numbered or bulleted lines are still usable choices.
    const lines = raw.split(/\r?\n/).filter((l) => LIST_MARKER.test(l))
    if (lines.length >= MIN_CHOICES) {
      obj = { choices: lines }
      warnings.push('The reply was not in the expected format; the choices were read from its list and no state changes were made.')
    } else return { ok: false, error: 'The reply could not be read.' }
  }
  const ended = obj.ended === true || (typeof obj.ended === 'string' && obj.ended.trim().toLowerCase() === 'true')
  const n = Math.min(MAX_CHOICES, Math.max(MIN_CHOICES, Math.round(choiceCount) || MIN_CHOICES))
  const choices = ended ? [] : cleanChoices(Array.isArray(obj.choices) ? obj.choices : typeof obj.choices === 'string' ? [obj.choices] : []).slice(0, n)
  if (!ended && !choices.length) return { ok: false, error: 'The reply had no choices.' }
  return { ok: true, value: { choices, ended, changes: normalizeChanges(obj.changes) }, warnings }
}

// ---------------------------------------------------------------------------------------------
// Cleaning the passage
// ---------------------------------------------------------------------------------------------

const PASSAGE_LABEL = /^\s*(?:#{1,6}\s*)?(?:\*\*|__)?passage(?:\s+\d+)?(?:\*\*|__)?(?:\s*[:\-–—.]\s*|[ \t]*\n\s*)(?:\*\*|__)?\s*/i
const CHOICE_HEADER = /^\W*(?:your\s+)?(?:choices|options|what\s+(?:do|will|would)\s+you\s+(?:do|choose)(?:\s+next)?)\W*$/i
const LIST_LINE = /^\s*(?:[-*•]+|\d+[.):]|[A-Da-d][.)])\s+\S/

/** Drops a list of choices the model wrote at the end of the passage anyway (with or without a header line). */
function stripChoiceList(text: string): string {
  const lines = text.split('\n')
  let end = lines.length
  while (end > 0 && !lines[end - 1].trim()) end--
  let start = end
  let items = 0
  while (start > 0) {
    const l = lines[start - 1]
    if (LIST_LINE.test(l)) {
      items++
      start--
    } else if (!l.trim() && items > 0) start--
    else break
  }
  if (items < MIN_CHOICES) return text
  // A header line right above the list ("Choices:", "What do you do?") goes with it.
  let head = start
  while (head > 0 && !lines[head - 1].trim()) head--
  if (head > 0 && CHOICE_HEADER.test(lines[head - 1].trim())) head--
  else head = start
  return lines.slice(0, head).join('\n').trim()
}

/** The passage as stored: preamble, fences and quotes stripped like Writing's prose, a leading "Passage:" label and a trailing choice list removed. */
export function cleanPassage(raw: string): string {
  let t = cleanProse(raw)
  for (let round = 0; round < 3; round++) {
    const before = t
    t = t.replace(PASSAGE_LABEL, '').trim()
    t = stripChoiceList(t).trim()
    // A whole passage in one pair of quotes is the model quoting its own story.
    t = t.replace(/^["“]([^"“”]*)["”]$/, '$1').trim()
    if (t === before) break
  }
  return t
}

// ---------------------------------------------------------------------------------------------
// Markdown export
// ---------------------------------------------------------------------------------------------

/** The path from the root to `nodeId` as one story. */
export function exportMarkdown(adv: Adventure, nodeId: string | null): string {
  const path = pathTo(adv, nodeId)
  const head = [`# ${adv.setup.title}`, '']
  const meta = [adv.setup.genre.trim(), adv.setup.world ? `From the Writing project ${adv.setup.world.title}` : ''].filter(Boolean).join(' · ')
  if (meta) head.push(`*${meta}*`, '')
  const parts: string[] = []
  path.forEach((n, i) => {
    if (i > 0) {
      if (n.input.kind === 'choice') parts.push(`*You chose: ${n.input.text.trim()}*`)
      else if (n.input.kind === 'action') parts.push(`*You: ${n.input.text.trim()}*`)
    }
    parts.push(n.text.trim())
  })
  if (path.length && path[path.length - 1].ended) parts.push('**The end.**')
  return `${head.join('\n')}\n${parts.join('\n\n')}\n`
}

const RESERVED_NAMES = /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])$/i

/** A file name for the export: the title without the characters Windows refuses, plus .md. */
export function exportFileName(title: string): string {
  // eslint-disable-next-line no-control-regex
  let base = title.replace(/[<>:"/\\|?*\u0000-\u001f]/g, ' ').replace(/\s+/g, ' ').trim()
  base = base.replace(/[. ]+$/, '').slice(0, 80).trim()
  if (!base) base = 'Adventure'
  if (RESERVED_NAMES.test(base)) base = `${base}_`
  return `${base}.md`
}

// ---------------------------------------------------------------------------------------------
// The world of a Writing project
// ---------------------------------------------------------------------------------------------

/** Most characters of world text copied into an adventure. */
export const WORLD_CAP = 12000
const OUTLINE_CAP = 3000

export interface WorldSource {
  /** Project title. */
  title: string
  /** Chapter titles with their summaries (what the project says happens); chapters without a summary are skipped. */
  chapters: { title: string; summary: string }[]
  bible: { name: string; kind: string; body: string }[]
}

const KIND_ORDER = ['character', 'place']

/**
 * Plain text for setup.world: what the project says the story is about, then characters, places and the other
 * entries, whole entries only (an entry that does not fit is skipped, a later short one may still fit). At most `cap`
 * characters.
 */
export function worldDigest(src: WorldSource, cap = WORLD_CAP): string {
  const parts: string[] = []
  let used = 0
  const add = (text: string): boolean => {
    const cost = text.length + (parts.length ? 2 : 0)
    if (used + cost > cap) return false
    parts.push(text)
    used += cost
    return true
  }
  const outline: string[] = []
  let outlineLen = 0
  for (const c of src.chapters) {
    const s = c.summary.trim()
    if (!s) continue
    const line = `- ${c.title.trim() || 'Chapter'}: ${s}`
    if (outlineLen + line.length > OUTLINE_CAP) break
    outline.push(line)
    outlineLen += line.length + 1
  }
  if (outline.length) add(`The project "${src.title}" so far:\n${outline.join('\n')}`)
  const rank = (kind: string): number => {
    const i = KIND_ORDER.indexOf(kind)
    return i === -1 ? KIND_ORDER.length : i
  }
  const entries = src.bible
    .map((e, i) => ({ e, i }))
    .sort((a, b) => rank(a.e.kind) - rank(b.e.kind) || a.i - b.i)
    .map((x) => x.e)
  let included = 0
  for (const e of entries) {
    const body = e.body.trim()
    const text = `## ${e.name.trim()} (${e.kind})\n${body}`.trim()
    if (add(text)) included++
  }
  // One entry bigger than the whole cap still gives the model something to go on.
  if (!included && entries.length) {
    const e = entries[0]
    const head = `## ${e.name.trim()} (${e.kind})\n`
    const room = cap - used - head.length - (parts.length ? 2 : 0) - 1
    if (room > 200) parts.push(`${head}${e.body.trim().slice(0, room).trimEnd()}…`)
  }
  return parts.join('\n\n')
}

// ---------------------------------------------------------------------------------------------
// Starter ideas
// ---------------------------------------------------------------------------------------------

export const ADVENTURE_STARTERS: AdventureStarter[] = [
  {
    id: 'starter-fantasy',
    title: 'The Lantern Road',
    genre: 'Fantasy',
    premise: 'The last lighthouse keeper of a drowned kingdom finds a map that leads to a lantern which, they say, can raise the sea floor. Three guilds want it, and so does something older.',
    character: 'Wren, a stubborn young lamplighter who has never left the coast',
    tone: 'Adventurous, with dry humour'
  },
  {
    id: 'starter-scifi',
    title: 'Quiet Ship',
    genre: 'Science fiction',
    premise: 'You wake from cold sleep on a colony ship that should still be a century from landing. The corridors are empty, the logs have been wiped, and something in the cargo hold is knocking.',
    character: 'Ilan Voss, the ship\'s junior engineer',
    tone: 'Tense and thoughtful'
  },
  {
    id: 'starter-noir',
    title: 'The Blue Envelope',
    genre: 'Mystery / noir',
    premise: 'A rainy night in a port city. A stranger leaves a blue envelope on your desk and a name written in your late partner\'s handwriting. By morning, two people who read it are dead.',
    character: 'Dana Marsh, a tired private detective with a bad knee and a good memory',
    tone: 'Hard-boiled, moody'
  },
  {
    id: 'starter-horror',
    title: 'The Guest Room',
    genre: 'Horror',
    premise: 'You inherit your grandmother\'s house in a village that has no road signs. The guest room is made up for someone who is expected tonight, and your grandmother left a note saying not to open the door for them.',
    character: 'Sam, a practical nurse who does not believe in ghosts',
    tone: 'Slow-burning dread'
  },
  {
    id: 'starter-cosy',
    title: 'The Bakery at the End of the Lane',
    genre: 'Cosy fantasy',
    premise: 'You have just taken over a small bakery in a village where the ovens are known to be a little magical. Today the first customer is a very polite dragon with a cold, and the village fair is in three days.',
    character: 'Poppy Quill, a new baker who is better at bread than at secrets',
    tone: 'Warm, gentle and funny'
  },
  {
    id: 'starter-pirate',
    title: 'The Salt-Eaten Crown',
    genre: 'Pirate adventure',
    premise: 'Your captain has died and left you a ship with a leaking hull, a crew that argues about everything, and a treasure map that only makes sense at low tide. A navy frigate is already on the horizon.',
    character: 'Mara "Quickknife" Delgado, first mate turned reluctant captain',
    tone: 'Swashbuckling and lively'
  },
  {
    id: 'starter-apocalypse',
    title: 'Beyond the Greenline',
    genre: 'Post-apocalyptic',
    premise: 'Thirty years after the collapse, your settlement\'s last well is running dry. A trader tells you of a valley past the Greenline where the water still runs clear, if you can get past the people who guard it.',
    character: 'Joss, a scavenger who knows every ruin within two days\' walk',
    tone: 'Gritty but hopeful'
  },
  {
    id: 'starter-heist',
    title: 'The Glass Vault',
    genre: 'Heist',
    premise: 'A museum that has never been robbed is about to open its new wing, and someone has hired you to take one painting. Your crew has one night, one blueprint with a mistake in it, and one member who is not who they say they are.',
    character: 'Nico Frey, the crew\'s planner, who always has a second plan',
    tone: 'Slick, clever, a little tongue-in-cheek'
  }
]
