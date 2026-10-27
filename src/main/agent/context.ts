// Keeps each request under ~80% of the model's context: older tool results are shortened first,
// and when that isn't enough, older turns are summarised. The system prompt and the current
// task are never dropped.
import type { WireMessage } from '@shared/types'

/** Code and command output run at roughly 3 characters per token; err on the safe side. */
export const CHARS_PER_TOKEN = 3.2
/** Share of the context a request may use; the rest is left for the reply. */
export const CONTEXT_SHARE = 0.8

export const estimateTokens = (text: string): number => Math.ceil(text.length / CHARS_PER_TOKEN)

export function messageTokens(m: WireMessage): number {
  const calls = m.tool_calls?.reduce((n, c) => n + c.function.name.length + c.function.arguments.length + 20, 0) ?? 0
  return estimateTokens(m.content) + Math.ceil(calls / CHARS_PER_TOKEN) + 8
}

export const totalTokens = (messages: WireMessage[]): number => messages.reduce((n, m) => n + messageTokens(m), 0)

/** Results at or below this size are always sent in full. */
export const COMPACT_MIN_CHARS = 800
const HEAD_CHARS = 500
const TAIL_CHARS = 200

/** Shortens a tool result to its beginning and end (adapted from RulZ LLM Suite). */
export function compactToolResult(text: string): string {
  if (text.length <= COMPACT_MIN_CHARS) return text
  let head = text.slice(0, HEAD_CHARS)
  const nl = head.lastIndexOf('\n')
  if (nl > HEAD_CHARS * 0.6) head = head.slice(0, nl)
  let tail = text.slice(-TAIL_CHARS)
  const tnl = tail.indexOf('\n')
  if (tnl >= 0 && tnl < TAIL_CHARS * 0.4) tail = tail.slice(tnl + 1)
  const omitted = text.length - head.length - tail.length
  return `${head}\n[… ${omitted.toLocaleString('en-US')} characters of this older result omitted to save space; run the tool again if you need them …]\n${tail}`
}

/** Compacts every tool result except the `keepRecent` most recent ones. */
export function compactOlderResults(messages: WireMessage[], keepRecent: number): WireMessage[] {
  const toolIdx = messages.flatMap((m, i) => (m.role === 'tool' ? [i] : []))
  const keep = new Set(keepRecent > 0 ? toolIdx.slice(-keepRecent) : [])
  return messages.map((m, i) => (m.role === 'tool' && !keep.has(i) ? { ...m, content: compactToolResult(m.content) } : m))
}

/** Tries progressively stronger compaction; `fits` is false when only summarising will help. */
export function fitMessages(
  messages: WireMessage[],
  budget: number,
  fixedTokens: number
): { messages: WireMessage[]; fits: boolean; tokens: number } {
  let tokens = fixedTokens + totalTokens(messages)
  if (tokens <= budget) return { messages, fits: true, tokens }
  let out = messages
  for (const keep of [6, 2, 0]) {
    out = compactOlderResults(messages, keep)
    tokens = fixedTokens + totalTokens(out)
    if (tokens <= budget) return { messages: out, fits: true, tokens }
  }
  return { messages: out, fits: false, tokens }
}

/**
 * Which messages to fold into the summary. Everything before the current task goes first;
 * if the task alone is too big, its older steps go, keeping the task message and the last
 * `keepTail` messages. Cuts land before an assistant message so calls and results stay paired.
 */
export function summaryRange(messages: WireMessage[], taskStart: number, keepTail = 6): { from: number; to: number } | null {
  if (taskStart > 0) return { from: 0, to: taskStart }
  let cut = messages.length - keepTail
  while (cut > 1 && messages[cut]?.role !== 'assistant') cut--
  return cut > 1 ? { from: 1, to: cut } : null
}

const clip = (s: string, n: number): string => (s.length > n ? `${s.slice(0, n)}…` : s)

/** Plain-text transcript of messages for the summariser, sized to fit `maxChars`. */
export function transcript(messages: WireMessage[], maxChars: number): string {
  const parts = messages.map((m) => {
    if (m.role === 'tool') return `RESULT: ${compactToolResult(m.content)}`
    const calls = m.tool_calls?.map((c) => `CALL ${c.function.name}(${clip(c.function.arguments, 400)})`).join('\n') ?? ''
    const who = m.role === 'user' ? 'OWNER' : m.role === 'assistant' ? 'AGENT' : m.role.toUpperCase()
    return `${who}: ${m.content}${calls ? `\n${calls}` : ''}`
  })
  let text = parts.join('\n\n')
  if (text.length > maxChars) text = `[…earlier part omitted…]\n${text.slice(-maxChars)}`
  return text
}

/** Summary without the model, used when the summary request fails. */
export function mechanicalSummary(messages: WireMessage[]): string {
  const lines: string[] = []
  for (const m of messages) {
    if (m.role === 'user') lines.push(`- The owner asked: ${clip(m.content.trim(), 300)}`)
    else if (m.role === 'assistant') {
      if (m.content.trim()) lines.push(`- Agent: ${clip(m.content.trim(), 200)}`)
      for (const c of m.tool_calls ?? []) lines.push(`  - ${c.function.name} ${clip(c.function.arguments, 120)}`)
    }
  }
  return clip(lines.join('\n'), 6000)
}

export const SUMMARY_PROMPT =
  'You summarise a coding session so the work can continue with less text. Write a compact summary of the ' +
  'transcript below: what the owner asked for, what was built (files and their purpose), commands that work ' +
  '(how to install, run and test), problems found and how they were fixed, and anything still unfinished. ' +
  'Keep exact file names, commands and error messages. Use short bullet points. Do not invent anything.'
