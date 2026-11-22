// Story bible entry files: a restricted YAML front matter + a Markdown body (design: "Bible entry file").
// Our own parser, not a YAML library: files are edited by hand in other apps, so it never throws and
// reports what it could not read. Lines it does not know are kept when the entry is written again.
import type { BibleEntry, BibleKind } from './novel'

const KINDS: BibleKind[] = ['character', 'place', 'lore', 'other']
const KNOWN_KEYS = ['name', 'kind', 'aliases', 'triggers', 'priority', 'always']

interface Block {
  /** Lower-cased key. */
  key: string
  /** The key line plus any indented / "- item" continuation lines, in the file's own text. */
  lines: string[]
  /** Text after the first ":" on the key line, comment removed. */
  value: string
  /** "- item" lines that follow an empty value (block list). */
  items: string[]
}

type Scalar = string | number | boolean

function splitFile(text: string): { front: string[] | null; body: string; unclosed: boolean } {
  const t = text.replace(/^﻿/, '').replace(/\r\n/g, '\n')
  if (!/^---[ \t]*\n/.test(t)) return { front: null, body: t, unclosed: false }
  const rest = t.slice(t.indexOf('\n') + 1)
  const lines = rest.split('\n')
  for (let i = 0; i < lines.length; i++) {
    if (/^---[ \t]*$/.test(lines[i])) {
      return { front: lines.slice(0, i), body: lines.slice(i + 1).join('\n'), unclosed: false }
    }
  }
  return { front: null, body: t, unclosed: true }
}

/** Removes a trailing " # comment" that is not inside quotes. */
function stripComment(s: string): string {
  let quote = ''
  // A quote only opens a quoted token at the start of a value or list item ("Mara's ring" is plain text).
  let tokenStart = true
  for (let i = 0; i < s.length; i++) {
    const c = s[i]
    if (quote) {
      if (c === '\\' && quote === '"') i++
      else if (c === quote) quote = ''
    } else if ((c === '"' || c === "'") && tokenStart) {
      quote = c
      tokenStart = false
    } else if (c === '#' && (i === 0 || /\s/.test(s[i - 1]))) return s.slice(0, i).trimEnd()
    else if (c === '[' || c === ',') tokenStart = true
    else if (!/\s/.test(c)) tokenStart = false
  }
  return s.trimEnd()
}

function toBlocks(front: string[]): Block[] {
  const blocks: Block[] = []
  let cur: Block | null = null
  for (const line of front) {
    const m = /^([A-Za-z_][\w-]*)[ \t]*:(.*)$/.exec(line)
    if (m && !/^\s/.test(line)) {
      cur = { key: m[1].toLowerCase(), lines: [line], value: stripComment(m[2].trim()), items: [] }
      blocks.push(cur)
    } else if (cur && (/^\s/.test(line) || /^-(\s|$)/.test(line) || line.trim() === '')) {
      cur.lines.push(line)
      const item = /^\s*-\s+(.*)$/.exec(line)
      if (item) cur.items.push(stripComment(item[1].trim()))
    } else if (line.trim() !== '') {
      // A comment or any other line at the left edge that is not "key: value": its own key-less block, so
      // formatEntry copies it back in place even when the keys around it are rewritten.
      blocks.push((cur = { key: '', lines: [line], value: '', items: [] }))
    }
  }
  // Blank lines at the very end of a block belong to the gap between blocks, not to the block.
  for (const b of blocks) while (b.lines.length > 1 && b.lines[b.lines.length - 1].trim() === '') b.lines.pop()
  return blocks
}

function parseScalar(raw: string): Scalar {
  const s = raw.trim()
  if (s.length >= 2 && s.startsWith('"') && s.endsWith('"')) {
    try {
      return JSON.parse(s) as string
    } catch {
      return s.slice(1, -1)
    }
  }
  if (s.length >= 2 && s.startsWith("'") && s.endsWith("'")) return s.slice(1, -1).replace(/''/g, "'")
  if (/^(true|yes)$/i.test(s)) return true
  if (/^(false|no)$/i.test(s)) return false
  if (/^-?\d+(\.\d+)?$/.test(s)) return Number(s)
  return s
}

/** "a, b, 'c, d'" → items; commas inside quotes stay. */
function splitList(inner: string): string[] {
  const out: string[] = []
  let cur = ''
  let quote = ''
  for (let i = 0; i < inner.length; i++) {
    const c = inner[i]
    if (quote) {
      cur += c
      if (c === '\\' && quote === '"' && i + 1 < inner.length) cur += inner[++i]
      else if (c === quote) quote = ''
    } else if ((c === '"' || c === "'") && cur.trim() === '') {
      // Only an item that starts with a quote is a quoted item; an apostrophe inside text is just text.
      quote = c
      cur += c
    } else if (c === ',') {
      out.push(cur)
      cur = ''
    } else cur += c
  }
  out.push(cur)
  return out.map((x) => x.trim()).filter((x) => x !== '')
}

function listOf(b: Block, problems: string[]): string[] {
  const v = b.value
  if (v.startsWith('[')) {
    if (!v.endsWith(']')) {
      problems.push(`"${b.key}" has an opening [ without a closing ].`)
      return splitList(v.slice(1)).map((x) => String(parseScalar(x)))
    }
    return splitList(v.slice(1, -1)).map((x) => String(parseScalar(x)))
  }
  if (v === '') return b.items.map((x) => String(parseScalar(x))).filter((x) => x !== '')
  // A single bare value: treat it as a one-item list.
  return [String(parseScalar(v))]
}

/** Reads a bible entry file. Never throws; anything it could not use is listed in `problems`. */
export function parseEntry(id: string, fileText: string): BibleEntry {
  const problems: string[] = []
  const { front, body, unclosed } = splitFile(fileText)
  if (unclosed) problems.push('The front matter is not closed with a --- line; the whole file is shown as the body.')
  const entry: BibleEntry = { id, name: id, kind: 'other', aliases: [], triggers: [], priority: 50, always: false, body, problems }
  if (!front) return entry
  const seen = new Set<string>()
  for (const b of toBlocks(front)) {
    if (!KNOWN_KEYS.includes(b.key)) continue
    if (seen.has(b.key)) {
      problems.push(`"${b.key}" appears more than once; the first one is used.`)
      continue
    }
    seen.add(b.key)
    switch (b.key) {
      case 'name': {
        const v = String(parseScalar(b.value))
        if (v) entry.name = v
        else problems.push('"name" is empty; the file name is used.')
        break
      }
      case 'kind': {
        const v = String(parseScalar(b.value)).toLowerCase()
        if ((KINDS as string[]).includes(v)) entry.kind = v as BibleKind
        else problems.push(`"kind" must be character, place, lore or other (found "${v}").`)
        break
      }
      case 'aliases':
        entry.aliases = listOf(b, problems)
        break
      case 'triggers':
        entry.triggers = listOf(b, problems)
        break
      case 'priority': {
        const v = parseScalar(b.value)
        if (typeof v === 'number' && Number.isFinite(v)) {
          entry.priority = Math.min(100, Math.max(0, Math.round(v)))
          if (v < 0 || v > 100) problems.push('"priority" must be between 0 and 100; it was clamped.')
        } else problems.push('"priority" must be a number from 0 to 100.')
        break
      }
      case 'always': {
        const v = parseScalar(b.value)
        if (typeof v === 'boolean') entry.always = v
        else problems.push('"always" must be true or false.')
        break
      }
    }
  }
  return entry
}

/** A string that reads back unchanged when written without quotes. */
function isPlain(s: string): boolean {
  if (s === '' || s !== s.trim() || /[\r\n,"']/.test(s)) return false
  if (/^[-"'#[\]{}&*!|>%@`]/.test(s) || /:(\s|$)/.test(s) || /\s#/.test(s)) return false
  return typeof parseScalar(s) === 'string'
}

/** A string as it should appear after "key: ": plain when that reads back the same, else quoted. */
function scalarText(s: string): string {
  return isPlain(s) ? s : JSON.stringify(s)
}

function listText(items: string[]): string {
  return `[${items.map((x) => (isPlain(x) && !x.includes(']') ? x : JSON.stringify(x))).join(', ')}]`
}

/**
 * Builds the file text: known keys first (name, kind, aliases, triggers, priority, always), then the
 * front-matter lines of `originalFileText` that this code does not know, in their original order, then
 * the body. Without an original there are no extra lines.
 */
export function formatEntry(entry: Omit<BibleEntry, 'id' | 'problems'>, originalFileText?: string): string {
  const lines = [
    `name: ${scalarText(entry.name)}`,
    `kind: ${entry.kind}`,
    `aliases: ${listText(entry.aliases)}`,
    `triggers: ${listText(entry.triggers)}`,
    `priority: ${Math.min(100, Math.max(0, Math.round(entry.priority)))}`,
    `always: ${entry.always ? 'true' : 'false'}`
  ]
  if (originalFileText !== undefined) {
    const { front } = splitFile(originalFileText)
    if (front) {
      for (const b of toBlocks(front)) {
        if (KNOWN_KEYS.includes(b.key)) continue
        lines.push(...b.lines)
      }
    }
  }
  return `---\n${lines.join('\n')}\n---\n${entry.body}`
}
