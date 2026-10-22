import type { ContextOverflow, WireMessage } from '@shared/types'

/** Rough token cost of one image for context accounting. */
export const IMAGE_TOKENS = 600

export function textOf(m: WireMessage): string {
  return typeof m.content === 'string'
    ? m.content
    : m.content.map((p) => (p.type === 'text' ? p.text : '')).join('\n')
}

export function imageCount(m: WireMessage): number {
  return typeof m.content === 'string' ? 0 : m.content.filter((p) => p.type === 'image_url').length
}

/** Messages that always stay: leading system messages plus the first user message. */
export function pinnedPrefixLength(messages: WireMessage[]): number {
  let i = 0
  while (i < messages.length && messages[i].role === 'system') i++
  return i < messages.length - 1 ? i + 1 : i
}

/**
 * Drops the oldest unpinned messages until `counts` (tokens per message) fit in `budget`.
 * The last message is always kept. Returns the kept indices.
 */
export function truncateMiddle(counts: number[], budget: number, pinned: number): number[] {
  const keep = counts.map((_, i) => i)
  let total = counts.reduce((a, b) => a + b, 0)
  let cursor = pinned
  while (total > budget && cursor < counts.length - 1) {
    total -= counts[cursor]
    keep.splice(keep.indexOf(cursor), 1)
    cursor++
  }
  return keep
}

export class ContextFullError extends Error {}

export interface ContextApi {
  countConversation: (messages: WireMessage[]) => Promise<number | null>
  countText: (text: string) => Promise<number | null>
}

/** Applies the overflow policy so the prompt plus `reserve` tokens fits in `nCtx`. */
export async function fitToContext(
  messages: WireMessage[],
  nCtx: number,
  reserve: number,
  policy: ContextOverflow,
  api: ContextApi
): Promise<WireMessage[]> {
  const budget = nCtx - reserve
  const total = await api.countConversation(messages)
  if (total === null || total <= budget) return messages
  if (policy === 'stop') {
    throw new ContextFullError(
      `This conversation needs about ${total} tokens but the model was loaded with a ${nCtx}-token context. ` +
        'Load the model with a larger context, start a new chat, or switch the overflow policy to "Truncate middle".'
    )
  }
  const counts = await Promise.all(
    messages.map(async (m) => ((await api.countText(textOf(m))) ?? textOf(m).length / 3) + 8 + imageCount(m) * IMAGE_TOKENS)
  )
  let kept = truncateMiddle(counts, budget, pinnedPrefixLength(messages)).map((i) => messages[i])
  // Per-message counts ignore template overhead; re-check and trim further if needed.
  for (let guard = 0; guard < 20; guard++) {
    const now = await api.countConversation(kept)
    if (now === null || now <= budget) break
    const pinned = pinnedPrefixLength(kept)
    if (pinned >= kept.length - 1) break
    kept = [...kept.slice(0, pinned), ...kept.slice(pinned + 1)]
  }
  return kept
}
