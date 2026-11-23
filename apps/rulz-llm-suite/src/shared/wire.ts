// Chat history to the wire format. Shared by the chat store (window) and the practice sessions the main process runs.
import type { ChatMessage, WireMessage } from './types'
import { compactToolResult, recentToolCallIds } from './tools'

/** Splits a leading <think>…</think> block out of content (same rule as the renderer's lib/thinking). */
function splitThink(content: string): { reasoning: string; content: string } | null {
  const m = content.match(/^\s*<think>([\s\S]*?)(<\/think>|$)/)
  if (!m) return null
  return { reasoning: m[1].trim(), content: content.slice(m[0].length).trimStart() }
}

/** The interrupted reply's thinking and visible answer, with inline <think> blocks split out. */
export function partialReply(m: ChatMessage): { reasoning: string; answer: string } {
  const split = splitThink(m.content)
  const reasoning = [m.reasoning?.trim(), split?.reasoning].filter(Boolean).join('\n\n')
  return { reasoning, answer: (split ? split.content : m.content).trim() }
}

/** What the model sees as its own interrupted turn. Never empty, so roles keep alternating. */
export function interruptedAssistantContent(m: ChatMessage): string {
  return partialReply(m).answer || '(I was still thinking when the user stopped me.)'
}

/**
 * The user's note as sent after an interrupted reply. Many chat templates drop the reasoning of
 * past assistant turns, so the partial thinking is quoted here in the user message, where every
 * template keeps it.
 */
export function noteAfterInterrupt(interrupted: ChatMessage, note: string): string {
  const { reasoning } = partialReply(interrupted)
  if (!reasoning) return `I stopped your reply to add something. Take this into account and carry on:\n\n${note}`
  const quoted = reasoning
    .split('\n')
    .map((l) => `> ${l}`)
    .join('\n')
  return `I stopped your reply to add something. This is what you had thought so far:\n\n${quoted}\n\nTake my note into account and carry on from there, without starting over:\n\n${note}`
}

/**
 * Converts chat history to the OpenAI message format llama-server expects. With `keepFull` set,
 * all but that many of the most recent tool results are shortened (the chat itself keeps them).
 */
export function buildWire(
  messages: ChatMessage[],
  systemPrompt: string,
  vision: boolean,
  keepFull: number | null,
  /** Send the thinking of assistant turns since the last user message back (interleaved thinking in a tool loop). */
  keepReasoning = false
): WireMessage[] {
  const keepIds = keepFull === null ? null : recentToolCallIds(messages, keepFull)
  let lastUser = -1
  messages.forEach((m, i) => {
    if (m.role === 'user') lastUser = i
  })
  const reasoningOf = (m: ChatMessage, i: number): { reasoning_content?: string } =>
    keepReasoning && i > lastUser && m.reasoning?.trim() ? { reasoning_content: m.reasoning } : {}
  const out: WireMessage[] = []
  if (systemPrompt.trim()) out.push({ role: 'system', content: systemPrompt })
  messages.forEach((m, i) => {
    const prev = messages[i - 1]
    if (m.role === 'assistant' && m.toolCalls?.length) {
      // A tool-calling turn becomes the assistant's calls followed by one result message per call.
      out.push({
        role: 'assistant',
        content: m.content,
        ...reasoningOf(m, i),
        tool_calls: m.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } }))
      })
      for (const c of m.toolCalls) {
        const result =
          c.result ?? (c.status === 'denied' ? 'The user declined to run this tool.' : 'The tool did not return a result.')
        const compact = keepIds !== null && !keepIds.has(c.id)
        out.push({ role: 'tool', tool_call_id: c.id, content: compact ? compactToolResult(result) : result })
      }
      return
    }
    if (m.role === 'assistant' && !m.content && m.error) return
    let text = m.role === 'assistant' && m.interrupted ? interruptedAssistantContent(m) : m.content
    if (m.role === 'user' && prev?.role === 'assistant' && prev.interrupted) text = noteAfterInterrupt(prev, text)
    for (const a of m.attachments ?? []) {
      if (a.kind === 'text' && a.text !== undefined) text += `\n\n\`\`\`${a.name}\n${a.text}\n\`\`\``
    }
    const images = vision ? (m.attachments ?? []).filter((a) => a.kind === 'image' && a.dataUrl) : []
    if (m.role === 'user' && images.length) {
      out.push({
        role: 'user',
        content: [
          { type: 'text', text },
          ...images.map((i) => ({ type: 'image_url' as const, image_url: { url: i.dataUrl! } }))
        ]
      })
    } else {
      out.push({ role: m.role, content: text })
    }
  })
  return out
}
