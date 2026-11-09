import type { ContextOverflow, WireMessage } from '@shared/types'

/** Rough token cost of one image for context accounting. */
export const IMAGE_TOKENS = 600

export function textOf(m: WireMessage): string {
  const text =
    typeof m.content === 'string' ? m.content : m.content.map((p) => (p.type === 'text' ? p.text : '')).join('\n')
  const calls = m.tool_calls?.map((c) => `${c.function.name}(${c.function.arguments})`).join('\n')
  // Thinking sent back to the template (RE sessions) takes context space too.
  const full = m.reasoning_content ? `${m.reasoning_content}\n${text}` : text
  return calls ? `${full}\n${calls}` : full
}

export function imageCount(m: WireMessage): number {
  return typeof m.content === 'string' ? 0 : m.content.filter((p) => p.type === 'image_url').length
}

/**
 * Splits the chat after the leading system messages into turns: a user message plus the
 * assistant and tool messages after it. Anything before the first user message (a greeting)
 * is a turn of its own. Returns each turn's message indices.
 */
export function chatTurns(messages: WireMessage[]): number[][] {
  let i = 0
  while (i < messages.length && messages[i].role === 'system') i++
  const turns: number[][] = []
  for (; i < messages.length; i++) {
    if (messages[i].role === 'user' || !turns.length) turns.push([i])
    else turns[turns.length - 1].push(i)
  }
  return turns
}

/** A turn's steps: each message on its own, except that tool results stay with their call. */
function turnSteps(messages: WireMessage[], turn: number[]): number[][] {
  const steps: number[][] = []
  for (const i of turn) {
    if (messages[i].role === 'tool' && steps.length) steps[steps.length - 1].push(i)
    else steps.push([i])
  }
  return steps
}

/**
 * What "Truncate middle" may remove, in the order it removes it. Whole turns go, so user and
 * assistant messages still alternate (Gemma 3's template raises an error otherwise) and every
 * tool call keeps its result:
 * 1. turns between the first and the current one, oldest first;
 * 2. the first turn, which usually sets up the task, once nothing else is left;
 * 3. older tool steps of the current turn, keeping its question and latest step, so a long tool
 *    loop can go on.
 * The system prompt, the current question and the latest step always stay; if they alone don't
 * fit, the request goes out as it is and llama-server reports the overflow.
 */
export function dropOrder(messages: WireMessage[]): number[][] {
  const turns = chatTurns(messages)
  if (!turns.length) return []
  const earlier = turns.slice(0, -1)
  const first = earlier.findIndex((t) => messages[t[0]].role === 'user')
  const units = earlier.filter((_, k) => k !== first)
  if (first >= 0) units.push(earlier[first])
  units.push(...turnSteps(messages, turns[turns.length - 1]).slice(1, -1))
  return units
}

/** Template tokens allowed per message (role markers, separators) and for the whole prompt. */
const MESSAGE_OVERHEAD = 16
const PROMPT_OVERHEAD = 64

/**
 * A count the real one can't exceed: byte-level and byte-fallback tokenizers never produce more
 * tokens than UTF-8 bytes, plus generous template overhead and the fixed image cost.
 */
export function upperBoundTokens(messages: WireMessage[]): number {
  return messages.reduce(
    (sum, m) => sum + Buffer.byteLength(textOf(m), 'utf8') + MESSAGE_OVERHEAD + imageCount(m) * IMAGE_TOKENS,
    PROMPT_OVERHEAD
  )
}

export class ContextFullError extends Error {}

export interface ContextApi {
  /** Tokens of the whole prompt as sent, tool definitions included. */
  countConversation: (messages: WireMessage[]) => Promise<number | null>
  countText: (text: string) => Promise<number | null>
  /** Tokens used by the tool definitions sent alongside the chat. */
  countTools?: () => Promise<number>
}

/** Rough cost of one message: its text, a few template tokens, and its images. */
async function messageTokens(m: WireMessage, api: ContextApi): Promise<number> {
  const text = textOf(m)
  const n = await api.countText(text).catch(() => null)
  return (n ?? text.length / 3) + 8 + imageCount(m) * IMAGE_TOKENS
}

/** How many removal units ahead to count while trimming, so counting overlaps a little. */
const COUNT_AHEAD = 4

/** Applies the overflow policy so the prompt plus `reserve` tokens fits in `nCtx`. */
export async function fitToContext(
  messages: WireMessage[],
  nCtx: number,
  reserve: number,
  policy: ContextOverflow,
  api: ContextApi
): Promise<WireMessage[]> {
  const budget = nCtx - reserve
  // Skip the template + tokenize round trip when the chat is clearly small enough.
  if (upperBoundTokens(messages) + ((await api.countTools?.()) ?? 0) <= budget) return messages
  const total = await api.countConversation(messages)
  if (total === null || total <= budget) return messages
  if (policy === 'stop') {
    throw new ContextFullError(
      `This conversation needs about ${total} tokens but the model was loaded with a ${nCtx}-token context. ` +
        'Load the model with a larger context, start a new chat, or switch the overflow policy to "Truncate middle".'
    )
  }
  const units = dropOrder(messages)
  // Only the messages that may go are counted, a few units ahead of where trimming has got to.
  const costs: Promise<number>[] = []
  const costOf = (k: number): Promise<number> => {
    for (let j = k; j < Math.min(units.length, k + COUNT_AHEAD); j++) {
      costs[j] ??= Promise.all(units[j].map((i) => messageTokens(messages[i], api))).then((n) => n.reduce((a, b) => a + b, 0))
    }
    return costs[k]
  }
  const dropped = new Set<number>()
  let next = 0
  // The measured total includes tool definitions and template overhead, so the overshoot is
  // what has to go; dropping a message frees at least its own count.
  let over = total - budget
  let kept = messages
  while (next < units.length) {
    let freed = 0
    while (next < units.length && freed < over) {
      freed += await costOf(next)
      for (const i of units[next++]) dropped.add(i)
    }
    kept = pairToolMessages(messages.filter((_, i) => !dropped.has(i)))
    if (next >= units.length) break
    const now = await api.countConversation(kept)
    if (now === null || now <= budget) break
    over = now - budget
  }
  return kept
}

/**
 * After trimming, keeps tool calls and tool results consistent: drops results whose call was
 * removed, and removes calls whose results were removed (the server rejects either).
 */
export function pairToolMessages(messages: WireMessage[]): WireMessage[] {
  const callIds = new Set(messages.flatMap((m) => m.tool_calls?.map((c) => c.id) ?? []))
  const resultIds = new Set(messages.filter((m) => m.role === 'tool').map((m) => m.tool_call_id))
  return messages
    .filter((m) => m.role !== 'tool' || callIds.has(m.tool_call_id ?? ''))
    .map((m) => {
      if (!m.tool_calls) return m
      const calls = m.tool_calls.filter((c) => resultIds.has(c.id))
      return calls.length === m.tool_calls.length ? m : { ...m, tool_calls: calls.length ? calls : undefined }
    })
}
