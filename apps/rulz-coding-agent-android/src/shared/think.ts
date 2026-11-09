/**
 * Splits a reply into the model's thinking and its answer, for models that write their reasoning inline
 * as <think>…</think> instead of sending it separately. Works on a partial reply while it streams:
 * an unclosed <think> means everything after it is still thinking. A lone </think> (the opening tag was
 * in the prompt) ends thinking that started at the beginning.
 */
export function splitThink(text: string): { reasoning: string; answer: string } {
  const parts: string[] = []
  let answer = text
  const lone = answer.indexOf('</think>')
  if (lone >= 0 && !answer.slice(0, lone).includes('<think>')) {
    parts.push(answer.slice(0, lone))
    answer = answer.slice(lone + 8)
  }
  answer = answer.replace(/<think>([\s\S]*?)(<\/think>|$)/g, (_, inner: string) => {
    parts.push(inner)
    return ''
  })
  return { reasoning: parts.map((p) => p.trim()).filter(Boolean).join('\n\n'), answer: answer.trim() }
}

/**
 * Whether a thinking box shows its thoughts. The owner's own click wins; otherwise it is open while the
 * model thinks, and stays open afterwards only when "Keep thinking open after the answer" is on.
 * Boxes that were never live (earlier tasks) start folded either way.
 */
export function thinkingOpen(chosen: boolean | null, live: boolean, wasLive: boolean, keepOpen: boolean): boolean {
  return chosen ?? (live || (keepOpen && wasLive))
}
