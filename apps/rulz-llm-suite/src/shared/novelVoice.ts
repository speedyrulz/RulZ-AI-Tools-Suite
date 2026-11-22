// Writing mode, stage 3: voice notes. A character entry's body may have a "## Voice" section (level 2 or 3,
// any case) up to the next heading of the same or a higher level. Pure and free of node: imports, because the
// renderer edits the entry buffer with these. Design: docs/writing-mode-design.md, "Voice notes".
import { splitParagraphs } from './novelText'

const NOTE_MAX = 500

interface Line {
  /** Line text without its line break. */
  text: string
  /** Offset of the first character of the line. */
  start: number
  /** Offset just after the line break (or the end of the text when the line has none). */
  end: number
  /** The line break that ends the line: '\r\n', '\n' or '' (last line). */
  eol: string
}

function toLines(text: string): Line[] {
  const lines: Line[] = []
  let start = 0
  while (start < text.length) {
    const lf = text.indexOf('\n', start)
    if (lf < 0) {
      lines.push({ text: text.slice(start), start, end: text.length, eol: '' })
      break
    }
    const crlf = lf > start && text[lf - 1] === '\r'
    lines.push({ text: text.slice(start, crlf ? lf - 1 : lf), start, end: lf + 1, eol: crlf ? '\r\n' : '\n' })
    start = lf + 1
  }
  return lines
}

const HEADING_LINE = /^ {0,3}(#{1,6})[ \t]+\S/
const VOICE_HEADING = /^ {0,3}(#{2,3})[ \t]+voice[ \t]*#*[ \t]*$/i

interface Section {
  /** Index of the heading line. */
  heading: number
  /** Index just after the section's last line. */
  endLine: number
}

function findVoiceSection(lines: Line[]): Section | null {
  for (let i = 0; i < lines.length; i++) {
    const match = VOICE_HEADING.exec(lines[i].text)
    if (!match) continue
    const level = match[1].length
    let end = i + 1
    while (end < lines.length) {
      const next = HEADING_LINE.exec(lines[end].text)
      if (next && next[1].length <= level) break
      end++
    }
    return { heading: i, endLine: end }
  }
  return null
}

/** The `## Voice` section's content (without its heading, trimmed); '' when there is none. */
export function voiceSection(body: string): string {
  const lines = toLines(body)
  const section = findVoiceSection(lines)
  if (!section) return ''
  return lines
    .slice(section.heading + 1, section.endLine)
    .map((line) => line.text)
    .join('\n')
    .trim()
}

/** Front matter and body of a whole entry file, by offsets, so nothing is re-written. */
function splitFront(fileText: string): { frontEnd: number; hasFront: boolean } {
  const bom = fileText.charCodeAt(0) === 0xfeff ? 1 : 0
  const open = /^---[ \t]*\r?\n/.exec(fileText.slice(bom))
  // Without front matter only the BOM is set aside, so it never sits in the body.
  if (!open) return { frontEnd: bom, hasFront: false }
  const lines = toLines(fileText.slice(bom + open[0].length))
  const base = bom + open[0].length
  for (const line of lines) {
    if (/^---[ \t]*$/.test(line.text)) return { frontEnd: base + line.end, hasFront: true }
  }
  // Not closed: the whole file is the body, as parseEntry reads it.
  return { frontEnd: bom, hasFront: false }
}

function majorityEol(text: string, fallback: string): string {
  const crlf = (text.match(/\r\n/g) ?? []).length
  const lf = (text.match(/\n/g) ?? []).length - crlf
  if (crlf === 0 && lf === 0) return fallback
  return crlf > lf ? '\r\n' : '\n'
}

/** One line, no leading #, at most 500 characters. '' when nothing is left. */
function cleanNote(note: string): string {
  let text = note.replace(/\s+/g, ' ').trim()
  text = text.replace(/^#+\s*/, '')
  if (text.length > NOTE_MAX) text = text.slice(0, NOTE_MAX).trimEnd()
  return text
}

/**
 * Adds `- <note>` to the voice section of an entry file (front matter + body, as the renderer holds it) and
 * returns the new text. Every other byte is kept: front matter, line endings, the rest of the body. A missing
 * section is created at the end of the body. An empty note returns the text unchanged.
 */
export function appendVoiceNote(fileText: string, note: string): string {
  const clean = cleanNote(note)
  if (!clean) return fileText
  const { frontEnd, hasFront } = splitFront(fileText)
  let front = fileText.slice(0, frontEnd)
  const body = fileText.slice(frontEnd)
  const frontEol = front ? majorityEol(front, '\n') : '\n'
  const eol = majorityEol(body, frontEol)
  const item = `- ${clean}`

  // The closing --- line of a file that holds nothing after it may lack its line break.
  if (body === '' && hasFront && !front.endsWith('\n')) front += eol

  const lines = toLines(body)
  const section = findVoiceSection(lines)
  if (section) {
    // After the last non-blank line of the section, so the note joins the list and the blank lines before the
    // next heading stay where they are.
    let last = section.heading
    for (let i = section.heading + 1; i < section.endLine; i++) if (lines[i].text.trim() !== '') last = i
    const at = lines[last].end
    // A last line without a line break (end of the file) gets one before the note.
    const lead = lines[last].eol === '' ? eol : ''
    return front + body.slice(0, at) + lead + item + eol + body.slice(at)
  }

  // No section: add one at the end, after a blank line.
  let out = body
  if (out !== '') {
    if (!/\n$/.test(out)) out += eol
    if (!/(\r?\n)[ \t]*\r?\n$/.test(out)) out += eol
  }
  return front + out + `## Voice${eol}${eol}${item}${eol}`
}

function cut(text: string, max: number): string {
  if (text.length <= max) return text
  if (max <= 1) return ''
  return `${text.slice(0, max - 1).trimEnd()}…`
}

// Cuts a block of lines to `max` characters at a line end where it can.
function cutLines(text: string, max: number): string {
  if (text.length <= max) return text
  const at = text.lastIndexOf('\n', max)
  if (at > 0) return text.slice(0, at).trimEnd()
  return cut(text, max)
}

/**
 * The entry body cut to its first paragraph plus its voice section, at most `maxChars` characters (the
 * voice section gets up to 60% of that). The short form the assembler uses for a voice target.
 */
export function shortWithVoice(body: string, maxChars = 1200): string {
  const lines = toLines(body)
  const section = findVoiceSection(lines)
  const voice = voiceSection(body)
  // The first paragraph of the body outside the voice section.
  const rest = section
    ? [...lines.slice(0, section.heading), ...lines.slice(section.endLine)].map((line) => line.text).join('\n')
    : body
  const first = (splitParagraphs(rest.trim())[0] ?? '').trim()
  if (!voice) return cut(first, maxChars)
  const heading = '## Voice\n'
  const voiceBudget = Math.min(heading.length + voice.length, Math.floor(maxChars * 0.6))
  // The voice text is cut on its own, so a long first note cannot swallow the line break after the heading.
  const voiceText = heading + cutLines(voice, Math.max(0, voiceBudget - heading.length))
  const firstBudget = Math.max(0, maxChars - voiceText.length - 2)
  const firstText = cut(first, firstBudget)
  return firstText ? `${firstText}\n\n${voiceText}` : voiceText
}
