// Small text helpers for the Writing mode. Pure and free of node: imports, because the renderer uses
// them too.

// A word is a run of letters/digits, optionally joined by apostrophes or hyphens ("Élan's",
// "well-made"). Markdown markers (#, *, ---) and bare punctuation never count.
const WORD = /[\p{L}\p{N}]+(?:['’‘-][\p{L}\p{N}]+)*/gu

export function wordCount(text: string): number {
  return (text.match(WORD) ?? []).length
}

// Blank lines (empty or whitespace only) separate blocks. Only the end of a block is trimmed, so
// indentation is kept; joining the blocks with a blank line restores the text up to whitespace.
export function splitParagraphs(text: string): string[] {
  return text
    .split(/\r?\n[ \t]*\r?\n(?:[ \t]*\r?\n)*/)
    .map((block) => block.trimEnd())
    .filter((block) => block.length > 0)
}

// Key stored in NovelChapter.summaryOf: a stable 53-bit hash (cyrb53) of the scene hashes. It is not
// secure and not meant to be; it only has to change when a scene hash does. No node:crypto, so it
// runs in the renderer as well. An empty list gives '' (= "no summary yet" in the outline).
export function sceneHashKey(hashes: string[]): string {
  if (hashes.length === 0) return ''
  // JSON keeps ['a,b'] and ['a', 'b'] apart, which a plain join would not.
  const text = JSON.stringify(hashes)
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
