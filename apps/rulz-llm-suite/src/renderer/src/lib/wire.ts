import type { ChatMessage, WireMessage } from '@shared/types'
import { compactToolResult, recentToolCallIds } from '@shared/tools'
import { interruptedAssistantContent, noteAfterInterrupt } from './interject'

/**
 * Converts chat history to the OpenAI message format llama-server expects. With `keepFull` set,
 * all but that many of the most recent tool results are shortened (the chat itself keeps them).
 */
export function buildWire(
  messages: ChatMessage[],
  systemPrompt: string,
  vision: boolean,
  keepFull: number | null
): WireMessage[] {
  const keepIds = keepFull === null ? null : recentToolCallIds(messages, keepFull)
  const out: WireMessage[] = []
  if (systemPrompt.trim()) out.push({ role: 'system', content: systemPrompt })
  messages.forEach((m, i) => {
    const prev = messages[i - 1]
    if (m.role === 'assistant' && m.toolCalls?.length) {
      // A tool-calling turn becomes the assistant's calls followed by one result message per call.
      out.push({
        role: 'assistant',
        content: m.content,
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
