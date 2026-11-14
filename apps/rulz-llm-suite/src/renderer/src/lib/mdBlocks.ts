const FENCE_OPEN = /^ {0,3}(`{3,}|~{3,})/
const FENCE_CLOSE = /^ {0,3}(`{3,}|~{3,})\s*$/
/** A $$ math block opens with 2+ dollars (and maybe a note without $) and closes on a line of dollars only. */
const MATH_OPEN = /^ {0,3}(\${2,})[^$]*$/
const MATH_CLOSE = /^ {0,3}(\${2,})[ \t]*$/
const LIST_ITEM = /^ {0,3}(?:[-*+]|\d{1,9}[.)])(?:[ \t]|$)/
/**
 * Link reference definitions and footnotes, also inside a quote or list item (`> [x]: url`): they
 * reach across blocks, so such text stays whole.
 */
const REF_DEF = /^[ \t]*(?:(?:>|[-*+]|\d{1,9}[.)])[ \t]*)*\[[^\]\n]+\]:/m
/** CommonMark HTML blocks of types 1-5 run to their end marker, blank lines included. */
const HTML_BLOCKS: [open: RegExp, close: RegExp][] = [
  [/^ {0,3}<(?:script|pre|style|textarea)(?=[\s>]|$)/i, /<\/(?:script|pre|style|textarea)>/i],
  [/^ {0,3}<!--/, /-->/],
  [/^ {0,3}<\?/, /\?>/],
  [/^ {0,3}<![a-zA-Z]/, />/],
  [/^ {0,3}<!\[CDATA\[/, /\]\]>/]
]

/**
 * Splits Markdown that is still streaming into blocks that render the same one by one as together,
 * so a memoized renderer only re-parses the last block on each update instead of the whole reply.
 * Splits only at blank lines outside code fences, $$ math and HTML blocks, never before an indented
 * line (it may belong to a list item) and never between list items (a loose list must stay one list).
 */
export function liveBlocks(src: string): string[] {
  if (!src.includes('\n\n') || REF_DEF.test(src)) return [src]
  const lines = src.split('\n')
  const blocks: string[] = []
  let start = 0
  let fence: { char: string; len: number } | null = null
  let math = 0 // dollars of the open math block
  let html: RegExp | null = null // end marker of the open HTML block
  let blankAt = -1 // first blank line of the run that may end the current block
  let blockHasList = false

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    if (fence) {
      const close = FENCE_CLOSE.exec(line)
      if (close && close[1][0] === fence.char && close[1].length >= fence.len) fence = null
      continue
    }
    if (math) {
      const close = MATH_CLOSE.exec(line)
      if (close && close[1].length >= math) math = 0
      continue
    }
    if (html) {
      if (html.test(line)) html = null
      continue
    }
    if (!line.trim()) {
      if (blankAt < 0) blankAt = i
      continue
    }
    if (blankAt >= 0) {
      const isList = LIST_ITEM.test(line)
      const indented = /^[ \t]/.test(line)
      if (!indented && !(isList && blockHasList)) {
        blocks.push(lines.slice(start, blankAt).join('\n'))
        start = i
        blockHasList = false
      }
      blankAt = -1
    }
    if (LIST_ITEM.test(line)) blockHasList = true
    const open = FENCE_OPEN.exec(line)
    if (open) {
      fence = { char: open[1][0], len: open[1].length }
      continue
    }
    const dollars = MATH_OPEN.exec(line)
    if (dollars) {
      math = dollars[1].length
      continue
    }
    for (const [opener, closer] of HTML_BLOCKS) {
      const m = opener.exec(line)
      if (m) {
        // The block may end on the line it starts on (`<!-- note -->`).
        if (!closer.test(line.slice(m[0].length))) html = closer
        break
      }
    }
  }
  blocks.push(lines.slice(start).join('\n'))
  return blocks
}
