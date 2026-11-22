// Writing mode, stage 3: consistency check helpers. Pure and free of node: imports (the renderer uses them).
// Design: docs/writing-mode-design.md, "Consistency check".
import type { CheckFinding } from './novel'

const SINGLE_QUOTES = /[‘’‚‛′ʼ]/
const DOUBLE_QUOTES = /[“”„‟″]/
const DASHES = /[‐‑‒–—―−]/
const SPACES = /\s/

interface Normalised {
  text: string
  /** For every character of `text`: offset of the original character it came from. */
  starts: number[]
  /** For every character of `text`: offset just after the original character it came from. */
  ends: number[]
}

/**
 * A copy for matching: curly quotes and dashes folded to plain ones, "…" to "...", every run of whitespace
 * one space, lower case. Leading and trailing whitespace is not trimmed here (the caller trims the quote).
 */
function normalise(text: string): Normalised {
  const out: string[] = []
  const starts: number[] = []
  const ends: number[] = []
  const push = (chars: string, start: number, end: number): void => {
    for (const ch of chars) {
      out.push(ch)
      starts.push(start)
      ends.push(end)
    }
  }
  let i = 0
  while (i < text.length) {
    const cp = text.codePointAt(i) as number
    const ch = String.fromCodePoint(cp)
    const next = i + ch.length
    if (SPACES.test(ch)) {
      if (out.length > 0 && out[out.length - 1] === ' ') ends[ends.length - 1] = next
      else push(' ', i, next)
    } else if (SINGLE_QUOTES.test(ch)) push("'", i, next)
    else if (DOUBLE_QUOTES.test(ch)) push('"', i, next)
    else if (DASHES.test(ch)) push('-', i, next)
    else if (ch === '…') push('...', i, next)
    else push(ch.toLowerCase(), i, next)
    i = next
  }
  return { text: out.join(''), starts, ends }
}

/** The quote as it is compared: normalised and trimmed. */
export function normaliseQuote(quote: string): string {
  return normalise(quote).text.trim()
}

/**
 * Finds `quote` in `text`: an exact match on normalised copies (case, curly quotes, dashes and whitespace
 * differences do not matter), first occurrence. Offsets are in the original text. null when not found.
 */
export function locateQuote(text: string, quote: string): { from: number; to: number } | null {
  const wanted = normaliseQuote(quote)
  if (!wanted) return null
  const haystack = normalise(text)
  const at = haystack.text.indexOf(wanted)
  if (at < 0) return null
  // `at` and the length are in UTF-16 units of the normalised text, which `starts` / `ends` index
  // per code point; they only differ when the text holds astral characters, so index through arrays.
  const units = [...haystack.text]
  const index = at === 0 ? 0 : [...haystack.text.slice(0, at)].length
  const length = [...wanted].length
  if (units.length !== haystack.starts.length) return null
  return { from: haystack.starts[index], to: haystack.ends[index + length - 1] }
}

/**
 * Stable short key of a finding for the dismissed list: a 53-bit hash (cyrb53) of its type and normalised
 * quote, 14 hex characters. Not secure; no node:crypto so it runs in the renderer too.
 */
export function findingKey(f: Pick<CheckFinding, 'type' | 'quote'>): string {
  const text = `${f.type}\n${normaliseQuote(f.quote)}`
  let h1 = 0xdeadbeef
  let h2 = 0x41c6ce57
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i)
    h1 = Math.imul(h1 ^ code, 2654435761)
    h2 = Math.imul(h2 ^ code, 1597334677)
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507)
  h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909)
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507)
  h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909)
  const value = 4294967296 * (2097151 & h2) + (h1 >>> 0)
  return value.toString(16).padStart(14, '0')
}
