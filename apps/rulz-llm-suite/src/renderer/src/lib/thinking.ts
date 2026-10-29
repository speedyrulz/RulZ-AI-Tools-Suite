/** Splits a leading <think>…</think> block out of content, for models whose reasoning isn't parsed server-side. */
export function splitThink(content: string): { reasoning: string; content: string; open: boolean } | null {
  const m = content.match(/^\s*<think>([\s\S]*?)(<\/think>|$)/)
  if (!m) return null
  return { reasoning: m[1].trim(), content: content.slice(m[0].length).trimStart(), open: !m[2] }
}

/**
 * Whether a reasoning block stays open once the model stops thinking. "Expand reasoning blocks by default"
 * and "Keep reasoning open after the answer" both leave it as it is (so a box the user folded stays folded);
 * otherwise it folds away.
 */
export function openAfterThinking(open: boolean, { defaultOpen, keepOpen }: { defaultOpen: boolean; keepOpen: boolean }): boolean {
  return defaultOpen || keepOpen ? open : false
}
