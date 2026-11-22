// Writing mode: prompt texts, reply schemas and parsers, project defaults. Pure; the renderer builds
// the text of each part here, counts it, and hands it to novelAssembly.ts.
import { z } from 'zod'
import { DEFAULT_SAMPLING } from './defaults'
import type {
  BibleEntry,
  CheckFinding,
  CheckReply,
  FindingType,
  NovelAction,
  NovelChapter,
  NovelProject,
  NovelScene,
  OutlineChaptersReply,
  OutlineScenesReply,
  RewriteVariant,
  StateKind,
  SummaryReply,
  UpdateStateChange,
  UpdateStateReply
} from './novel'
import { NOVEL_FORMAT } from './novel'
import { splitParagraphs } from './novelText'
import type { SamplingParams } from './types'

// ---------------------------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------------------------

/** Sampling for new projects: open, varied prose; DRY and XTC against repeats and stock phrases. */
export const FICTION_SAMPLING: SamplingParams = {
  ...DEFAULT_SAMPLING,
  temperature: 0.9,
  minP: 0.05,
  topP: 1,
  topK: 0,
  repeatPenalty: 1,
  dryMultiplier: 0.8,
  xtcProbability: 0.5,
  xtcThreshold: 0.1,
  thinking: 'off'
}

const ID_ALPHABET = '0123456789abcdefghijklmnopqrstuvwxyz'

// Local on purpose: novelOutline.ts has its own newId, but this file must not depend on it.
function randomId(prefix: string): string {
  let id = ''
  for (let i = 0; i < 6; i++) id += ID_ALPHABET[Math.floor(Math.random() * ID_ALPHABET.length)]
  return `${prefix}-${id}`
}

export function defaultProject(title: string): NovelProject {
  const now = new Date().toISOString()
  return {
    format: NOVEL_FORMAT,
    id: randomId('p'),
    title,
    createdAt: now,
    updatedAt: now,
    modelId: null,
    // A copy, so changing one project's lists never touches the shared preset.
    sampling: {
      ...FICTION_SAMPLING,
      drySequenceBreakers: [...FICTION_SAMPLING.drySequenceBreakers],
      avoidWords: [...FICTION_SAMPLING.avoidWords],
      stop: [...FICTION_SAMPLING.stop]
    },
    adapters: null,
    budget: { replyTokens: 1024, proseShare: 0.35, bibleShare: 0.3 },
    povTense: ''
  }
}

// ---------------------------------------------------------------------------------------------
// Prompt texts
// ---------------------------------------------------------------------------------------------

const PROSE_ACTIONS: NovelAction[] = ['continue', 'rewrite', 'expand', 'condense', 'dialogue', 'draftScene']

export function isProseAction(action: NovelAction): boolean {
  return PROSE_ACTIONS.includes(action)
}

export function systemText(action: NovelAction, style: string, povTense: string): string {
  const parts = ['You are a fiction writer working on a novel with the author.']
  if (style.trim()) parts.push(`The author's style notes:\n${style.trim()}`)
  if (povTense.trim()) parts.push(`Point of view and tense: ${povTense.trim()}`)
  if (isProseAction(action)) {
    // Continue and dialogue carry on from the text; the other actions work on given text or a new scene.
    const voice =
      action === 'continue' || action === 'dialogue'
        ? 'Continue in the same voice as the text so far.'
        : 'Match the voice of the style notes and the text around it.'
    parts.push(
      'Write only the story text: no commentary, no headings, no quotation marks around it. ' +
        `${voice} Stay within the beats of the scene and do not resolve the scene early.`
    )
  } else {
    parts.push('Answer only in the JSON format the task asks for, with no commentary.')
  }
  return parts.join('\n\n')
}

const REWRITE_VARIANTS: Record<RewriteVariant, string> = {
  tighter:
    'Rewrite the selected passage so it is tighter: cut padding and repeated ideas, ' +
    'keep the meaning and the voice.',
  vivid:
    'Rewrite the selected passage so it is more vivid: concrete, specific detail and stronger verbs, ' +
    'keeping the meaning and the voice.',
  tone: "Rewrite the selected passage in a different tone, given in the author's note. Keep what happens.",
  show:
    'Rewrite the selected passage to show instead of tell: replace stated feelings and summary ' +
    'with action, sensation and dialogue.',
  custom: "Rewrite the selected passage as the author's note asks."
}

// Tone and custom depend on the author's note; without one they still need a usable instruction.
function rewriteInstruction(variant: RewriteVariant, hasNote: boolean): string {
  if (variant === 'tone' && !hasNote) {
    return (
      'Rewrite the selected passage in a noticeably different tone ' +
      '(calmer, darker or lighter than now). Keep what happens.'
    )
  }
  if (variant === 'custom' && !hasNote) {
    return 'Rewrite the selected passage to improve it, keeping its meaning and the voice.'
  }
  return REWRITE_VARIANTS[variant]
}

export interface TaskOptions {
  variant?: RewriteVariant
  note?: string
  selection?: string
  targetWords?: number | null
  /** Outline actions: how many chapters or scenes. Alternatives for prose actions are not in the prompt. */
  count?: number
  /** updateState: the state before the scene, one fact per line with its id, e.g. "[f1a2b3] Mara Venn - location: the lighthouse". */
  stateBeforeText?: string
  /** updateState: facts already added or ended at this scene, same format. */
  recordedText?: string
}

export function taskText(action: NovelAction, opts: TaskOptions): string {
  const words = opts.targetWords && opts.targetWords > 0 ? `about ${Math.round(opts.targetWords)} words` : ''
  const count = opts.count && opts.count > 0 ? String(Math.round(opts.count)) : ''
  let task: string
  switch (action) {
    case 'continue':
      task =
        'Continue the text from the exact point where it ends' +
        (words ? `, for ${words}` : '') +
        '. If the last sentence is unfinished, finish it. Do not repeat the last sentence or retell what came before.'
      break
    case 'rewrite':
      task = `${rewriteInstruction(opts.variant ?? 'tighter', Boolean(opts.note?.trim()))} Output only the replacement text.`
      break
    case 'expand':
      task =
        'Expand the selected passage with more detail, action or interior thought, keeping its meaning and the voice.' +
        (words ? ` Aim for ${words}.` : '') +
        ' Output only the replacement text.'
      break
    case 'condense':
      task =
        'Condense the selected passage to its essentials, keeping its meaning and the voice.' +
        (words ? ` Aim for ${words}.` : '') +
        ' Output only the replacement text.'
      break
    case 'dialogue':
      task =
        'Write the dialogue for the current beat, continuing from where the text ends. ' +
        'Keep each character in their own voice. Output only the dialogue and the short action tags that go with it.'
      break
    case 'draftScene':
      task =
        'Draft the scene from its beats' +
        (words ? `, ${words}` : '') +
        '. Follow the beats in order, start at the beginning of the scene and stop where the last beat ends.'
      break
    case 'summarizeScene':
      task =
        'Summarize the scene as a recap for later chapters, in at most 200 words and in past tense. ' +
        "Say what happens and what changes in the characters' situation (what they learn, lose or decide). " +
        'Reply as JSON: {"summary": "..."}.'
      break
    case 'summarizeChapter':
      task =
        'Summarize the chapter from its scene summaries, in at most 200 words and in past tense. ' +
        "Say what happens and what changes in the characters' situation. " +
        'Reply as JSON: {"summary": "..."}.'
      break
    case 'outlineChapters':
      task =
        `Outline ${count ? `${count} chapters` : 'the chapters'} for the novel from the premise and the style notes. ` +
        'Give each chapter a title, an act label (such as "Act I") and a summary of one to three sentences. ' +
        'Reply as JSON: {"chapters": [{"title": "...", "act": "...", "summary": "..."}]}.'
      break
    case 'outlineScenes':
      task =
        `Outline ${count ? `${count} scenes` : 'the scenes'} for this chapter. ` +
        'Give each scene a title and beats (two to five sentences of what happens). ' +
        'For pov and present use the ids of the bible entries listed in the prompt (the text after "id:"); ' +
        'use null for pov when no entry fits. ' +
        'Reply as JSON: {"scenes": [{"title": "...", "beats": "...", "pov": null, "present": []}]}.'
      break
    case 'checkScene':
      task =
        'Check the scene for inconsistencies with the style notes, the story bible, the story so far and the ' +
        'state before the scene. Look for: continuity (facts, objects, places, who knows what), character ' +
        '(behaviour or voice that does not fit the entry), timeline, point of view slips, and tense slips. ' +
        'Report only real problems, at most 20, most serious first. For each, "quote" is the exact words from the ' +
        'scene (copy them, a short run of words, not a whole paragraph), "problem" says what is wrong, ' +
        '"conflictsWith" says what it contradicts, and "suggestion" is a short fix. "type" is one of ' +
        'continuity, character, timeline, pov, tense, other. If there are no problems, reply with an empty list. ' +
        'Reply as JSON: {"findings": [{"type": "continuity", "quote": "...", "problem": "...", ' +
        '"conflictsWith": "...", "suggestion": "..."}]}.'
      break
    case 'updateState':
      task =
        'Update the tracked state from the scene. Report what became true in this scene and what stopped being ' +
        'true in it, as the characters\' situation changes (where they are, what they know or carry, their ' +
        'condition, how they stand with each other). Only report changes the scene itself shows; do not repeat ' +
        'anything in the state before the scene or already recorded for this scene. ' +
        '"add" gives entity (the name of a bible entry), kind (location, knows, carries, condition, relationship ' +
        'or other), value (one short line) and about (the other entry\'s name for a relationship, otherwise null). ' +
        '"end" gives the factId from the state before the scene, in brackets, and a short reason. ' +
        'If nothing changed, reply with an empty list. ' +
        'Reply as JSON: {"changes": [{"op": "add", "entity": "...", "kind": "location", "value": "...", ' +
        '"about": null}, {"op": "end", "factId": "...", "reason": "..."}]}.'
      break
    case 'interview':
      // The interview is a chat with its own system prompt (interviewSystem); there is no task text.
      task = "Answer the author's question in character."
      break
  }
  const sections = [task]
  if (action === 'updateState') {
    sections.push(`State before this scene (fact ids in brackets):\n${opts.stateBeforeText?.trim() || '(none)'}`)
    sections.push(`Already recorded at this scene:\n${opts.recordedText?.trim() || '(none)'}`)
  }
  const selection = opts.selection?.trim()
  if (selection) sections.push(`Selected passage:\n${selection}`)
  const note = opts.note?.trim()
  if (note) sections.push(`The author's note: ${note}`)
  return sections.join('\n\n')
}

export function sceneText(scene: NovelScene, chapter: NovelChapter, entriesById: Map<string, BibleEntry>): string {
  // An id that is not in the bible any more is still shown, so the model sees the gap.
  const nameOf = (id: string): string => entriesById.get(id)?.name ?? id
  const lines = [`Chapter: ${chapter.title}`, `Scene: ${scene.title}`]
  if (scene.pov) lines.push(`POV: ${nameOf(scene.pov)}`)
  if (scene.present.length > 0) lines.push(`Present: ${scene.present.map(nameOf).join(', ')}`)
  if (scene.targetWords !== null && scene.targetWords > 0) lines.push(`Target length: ${scene.targetWords} words`)
  if (scene.beats.trim()) lines.push(`Beats:\n${scene.beats.trim()}`)
  return lines.join('\n')
}

function entryWithBody(entry: BibleEntry, body: string): string {
  const head = [`### ${entry.name} (${entry.kind})`]
  if (entry.aliases.length > 0) head.push(`Aliases: ${entry.aliases.join(', ')}`)
  const text = head.join('\n')
  return body ? `${text}\n\n${body}` : text
}

/** How a bible entry is shown to the model. */
export function entryText(entry: BibleEntry): string {
  return entryWithBody(entry, entry.body.trim())
}

/** The same with the first paragraph of the body only: the assembler uses it before dropping an entry. */
export function entryShortText(entry: BibleEntry): string {
  return entryWithBody(entry, splitParagraphs(entry.body.trim())[0] ?? '')
}

/** One line per entry for the outline prompts, with the id the model must use in pov and present. */
export function entryListText(entries: BibleEntry[]): string {
  return entries.map((entry) => `- ${entry.name} (${entry.kind}, id: ${entry.id})`).join('\n')
}

/**
 * Maps a pov/present value from a model reply to an entry id: the id exactly, then a name or alias
 * ignoring case and surrounding space. null when nothing matches (the renderer drops the reference).
 */
export function resolveEntryRef(entries: BibleEntry[], ref: string): string | null {
  const wanted = ref.trim()
  if (!wanted) return null
  const byId = entries.find((entry) => entry.id === wanted)
  if (byId) return byId.id
  const lower = wanted.toLowerCase()
  const byName = entries.find((entry) =>
    [entry.name, ...entry.aliases].some((value) => value.trim().toLowerCase() === lower)
  )
  return byName ? byName.id : null
}

export function summaryText(label: string, summary: string): string {
  return `### ${label}\n${summary.trim()}`
}

// ---------------------------------------------------------------------------------------------
// Structured replies
// ---------------------------------------------------------------------------------------------

const STRING = { type: 'string' }

export const OUTLINE_CHAPTERS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['chapters'],
  properties: {
    chapters: {
      type: 'array',
      minItems: 1,
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'act', 'summary'],
        properties: { title: STRING, act: STRING, summary: STRING }
      }
    }
  }
}

export const OUTLINE_SCENES_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['scenes'],
  properties: {
    scenes: {
      type: 'array',
      minItems: 1,
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'beats', 'pov', 'present'],
        properties: {
          title: STRING,
          beats: STRING,
          pov: { type: ['string', 'null'] },
          present: { type: 'array', items: STRING }
        }
      }
    }
  }
}

export const SUMMARY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['summary'],
  properties: { summary: STRING }
}

const chaptersReply = z.object({
  chapters: z
    .array(
      z.object({
        title: z.string().trim().min(1, 'A chapter has no title'),
        act: z.string().trim(),
        summary: z.string().trim().min(1, 'A chapter has no summary')
      })
    )
    .min(1, 'The reply has no chapters')
})

const scenesReply = z.object({
  scenes: z
    .array(
      z.object({
        title: z.string().trim().min(1, 'A scene has no title'),
        beats: z.string().trim().min(1, 'A scene has no beats'),
        // Models sometimes write "" for "nobody".
        pov: z.string().nullable().transform((value) => (value && value.trim() ? value.trim() : null)),
        // Blank entries are dropped.
        present: z.array(z.string()).transform((list) => list.map((id) => id.trim()).filter(Boolean))
      })
    )
    .min(1, 'The reply has no scenes')
})

const summaryReply = z.object({ summary: z.string().trim().min(1, 'The summary is empty') })

type Parsed<T> = { ok: true; value: T } | { ok: false; error: string }

// The JSON may be the whole reply, inside a ```json fence, or after some leading text. Try those in
// that order and keep the first that parses.
function jsonCandidates(raw: string): string[] {
  const candidates: string[] = [raw.trim()]
  const fence = raw.match(/```(?:json)?[ \t]*\r?\n?([\s\S]*?)```/i)
  if (fence) candidates.push(fence[1].trim())
  const start = raw.indexOf('{')
  const end = raw.lastIndexOf('}')
  if (start >= 0 && end > start) candidates.push(raw.slice(start, end + 1))
  return candidates
}

function parseReply<T>(raw: string, schema: z.ZodType<T>): Parsed<T> {
  let firstError = ''
  for (const candidate of jsonCandidates(raw)) {
    let json: unknown
    try {
      json = JSON.parse(candidate)
    } catch {
      continue
    }
    const result = schema.safeParse(json)
    if (result.success) return { ok: true, value: result.data }
    // The text was JSON but the wrong shape: that is the more useful error.
    const issue = result.error.issues[0]
    const where = issue.path.length > 0 ? `${issue.path.join('.')}: ` : ''
    firstError ||= `The reply has the wrong shape (${where}${issue.message}).`
  }
  return { ok: false, error: firstError || 'The reply is not valid JSON.' }
}

export function parseOutlineChapters(raw: string): Parsed<OutlineChaptersReply> {
  return parseReply(raw, chaptersReply)
}

export function parseOutlineScenes(raw: string): Parsed<OutlineScenesReply> {
  return parseReply(raw, scenesReply)
}

export function parseSummary(raw: string): Parsed<SummaryReply> {
  return parseReply(raw, summaryReply)
}

// ---------------------------------------------------------------------------------------------
// Stage 3: checks and tracked state
// ---------------------------------------------------------------------------------------------

const FINDING_TYPES: FindingType[] = ['continuity', 'character', 'timeline', 'pov', 'tense', 'other']
const STATE_KINDS: StateKind[] = ['location', 'knows', 'carries', 'condition', 'relationship', 'other']
const MAX_FINDINGS = 20
const VALUE_MAX = 300

export const CHECK_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['findings'],
  properties: {
    findings: {
      type: 'array',
      minItems: 0,
      maxItems: MAX_FINDINGS,
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['type', 'quote', 'problem', 'conflictsWith', 'suggestion'],
        properties: {
          type: { type: 'string', enum: FINDING_TYPES },
          quote: STRING,
          problem: STRING,
          conflictsWith: STRING,
          suggestion: STRING
        }
      }
    }
  }
}

export const UPDATE_STATE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['changes'],
  properties: {
    changes: {
      type: 'array',
      minItems: 0,
      items: {
        anyOf: [
          {
            type: 'object',
            additionalProperties: false,
            required: ['op', 'entity', 'kind', 'value', 'about'],
            properties: {
              op: { type: 'string', enum: ['add'] },
              entity: STRING,
              kind: { type: 'string', enum: STATE_KINDS },
              value: STRING,
              about: { type: ['string', 'null'] }
            }
          },
          {
            type: 'object',
            additionalProperties: false,
            required: ['op', 'factId', 'reason'],
            properties: {
              op: { type: 'string', enum: ['end'] },
              factId: STRING,
              reason: STRING
            }
          }
        ]
      }
    }
  }
}

// Replies are read leniently item by item: a bad item is left out, it does not sink the others.
const lenientReply = (key: string): z.ZodType<unknown[]> =>
  z.object({ [key]: z.array(z.unknown()) }).transform((value) => value[key] as unknown[])

const str = (value: unknown): string => (typeof value === 'string' ? value : '')
const oneLine = (value: unknown, max: number): string => str(value).replace(/\s+/g, ' ').trim().slice(0, max).trim()
const record = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null

function findingOf(item: unknown): CheckFinding | null {
  const obj = record(item)
  if (!obj) return null
  const type = str(obj.type).trim().toLowerCase() as FindingType
  if (!FINDING_TYPES.includes(type)) return null
  const quote = str(obj.quote).trim()
  const problem = str(obj.problem).trim()
  if (!quote || !problem) return null
  return { type, quote, problem, conflictsWith: str(obj.conflictsWith).trim(), suggestion: str(obj.suggestion).trim() }
}

function changeOf(item: unknown): UpdateStateChange | null {
  const obj = record(item)
  if (!obj) return null
  const op = str(obj.op).trim().toLowerCase()
  if (op === 'add') {
    const kind = str(obj.kind).trim().toLowerCase() as StateKind
    if (!STATE_KINDS.includes(kind)) return null
    const entity = str(obj.entity).trim()
    const value = oneLine(obj.value, VALUE_MAX)
    if (!entity || !value) return null
    // 'about' means something for a relationship only.
    const about = kind === 'relationship' ? str(obj.about).trim() || null : null
    return { op: 'add', entity, kind, value, about }
  }
  if (op === 'end') {
    const factId = str(obj.factId).trim().replace(/^\[+|\]+$/g, '').trim()
    if (!factId) return null
    return { op: 'end', factId, reason: oneLine(obj.reason, VALUE_MAX) }
  }
  return null
}

/** Like parseCheckReply but with the error instead of a throw. */
export function parseCheckReplyResult(raw: string): Parsed<CheckReply> {
  const parsed = parseReply(raw, lenientReply('findings'))
  if (!parsed.ok) return parsed
  const findings = parsed.value.map(findingOf).filter((finding): finding is CheckFinding => finding !== null)
  return { ok: true, value: { findings: findings.slice(0, MAX_FINDINGS) } }
}

/** At most 20 findings, values trimmed, unknown types and empty quote/problem dropped. Throws when the reply is not the expected JSON. */
export function parseCheckReply(text: string): CheckReply {
  const result = parseCheckReplyResult(text)
  if (!result.ok) throw new Error(result.error)
  return result.value
}

/** Like parseUpdateStateReply but with the error instead of a throw. */
export function parseUpdateStateReplyResult(raw: string): Parsed<UpdateStateReply> {
  const parsed = parseReply(raw, lenientReply('changes'))
  if (!parsed.ok) return parsed
  const changes = parsed.value.map(changeOf).filter((change): change is UpdateStateChange => change !== null)
  return { ok: true, value: { changes } }
}

/** Unknown ops and kinds dropped, values one line and at most 300 characters. Throws when the reply is not the expected JSON. */
export function parseUpdateStateReply(text: string): UpdateStateReply {
  const result = parseUpdateStateReplyResult(text)
  if (!result.ok) throw new Error(result.error)
  return result.value
}

/** The system prompt of a character interview (stage 3). */
export function interviewSystem(input: { name: string; body: string; stateLines: string; summaries: string[] }): string {
  const name = input.name.trim() || 'the character'
  const parts = [
    `You are ${name}. Answer in character, in your own voice. You know only what ${name} knows by this point in the story; if asked about later events, you do not know them. Stay in character: do not mention that you are a character in a novel.`
  ]
  const body = input.body.trim()
  if (body) {
    parts.push(
      "The author's notes about you (they may mention things you have not lived yet; do not reveal those):\n" + body
    )
  }
  const state = input.stateLines.trim()
  if (state) parts.push(`What is true of you at this point in the story:\n${state}`)
  const summaries = input.summaries.map((summary) => summary.trim()).filter(Boolean)
  if (summaries.length > 0) {
    parts.push(`What has happened so far, in the scenes you were in:\n${summaries.join('\n\n')}`)
  }
  return parts.join('\n\n')
}

// ---------------------------------------------------------------------------------------------
// Cleaning prose replies
// ---------------------------------------------------------------------------------------------

// "Sure, here is the passage:" / "Certainly!" on a line of their own. "Sure enough, it rained." and
// "Sure." are story text and do not match: after Sure/Certainly only "here is ..." or ! / : may follow.
const PREAMBLE_LINE =
  /^(?:(?:sure|certainly|of course|absolutely)(?:[,!:.]?[ \t]+here(?:'s|\u2019s| is| are)\b.*|[!:])|here(?:'s|\u2019s| is| are)\b.*:)[ \t]*$/i
// "(Word count: 312)", "*Note: ...*", "Let me know if ...": only on a last line of their own.
const TRAILING_NOTE =
  /^(?:(?:[([*_]+\s*)?word count\b|[([*_]+\s*note\b|let me know\b|i hope this (?:helps|meets|is what|captures|works)\b|feel free to\b|would you like me to\b)[^\n]*$/i
const QUOTE_CHARS = /["“”]/

function stripPreamble(text: string): string {
  const newline = text.indexOf('\n')
  if (newline < 0) return text
  if (!PREAMBLE_LINE.test(text.slice(0, newline))) return text
  return text.slice(newline + 1)
}

function stripFence(text: string): string {
  const match = text.match(/^```[^\n]*\n([\s\S]*?)\n?```$/)
  return match ? match[1] : text
}

function stripRules(text: string): string {
  return text.replace(/^(?:-{3,}[ \t]*\r?\n)+/, '').replace(/(?:\r?\n[ \t]*-{3,})+[ \t]*$/, '')
}

function stripTrailingNote(text: string): string {
  const newline = text.lastIndexOf('\n')
  if (newline < 0) return text
  return TRAILING_NOTE.test(text.slice(newline + 1).trim()) ? text.slice(0, newline) : text
}

// A whole passage wrapped in one pair of quotes loses them, but only when it spans several lines or
// paragraphs and has no quotes inside. A single line or paragraph is never unquoted: it is far more
// likely to be someone speaking than a model wrapping its answer.
function stripWrappingQuotes(text: string): string {
  if (text.length < 2) return text
  const first = text[0]
  const last = text[text.length - 1]
  if (!((first === '"' && last === '"') || (first === '“' && last === '”'))) return text
  const inner = text.slice(1, -1)
  if (QUOTE_CHARS.test(inner)) return text
  return inner.includes('\n') ? inner.trim() : text
}

export function cleanProse(raw: string): string {
  // CRLF would leave a \r in front of every line end and defeat the line patterns.
  let text = raw.replace(/\r\n?/g, '\n').trim()
  // Peel layers until nothing changes: a reply can be a preamble, then a fence, then a note.
  for (let round = 0; round < 6; round++) {
    const before = text
    text = stripPreamble(text).trim()
    text = stripFence(text).trim()
    text = stripRules(text).trim()
    text = stripTrailingNote(text).trim()
    text = stripRules(text).trim()
    text = stripWrappingQuotes(text)
    if (text === before) break
  }
  return text
}
