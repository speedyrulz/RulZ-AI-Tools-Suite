// Reads the reply of a proxied /v1/chat/completions request (streamed or not) as it passes through,
// so RE sessions run by other apps can be recorded like in-app ones.
import type { WireToolCall } from '@shared/types'
import type { RecordedRound } from '../chat/service'

type Reply = RecordedRound['response']

interface ToolDelta {
  index?: number
  id?: string
  function?: { name?: string; arguments?: string }
}

export class ChatReplyCapture {
  private buf = ''
  private raw: Buffer[] = []
  private content = ''
  private reasoning = ''
  private finishReason: string | undefined
  private calls = new Map<number, { id: string; name: string; args: string }>()
  private decoder = new TextDecoder()

  constructor(private readonly sse: boolean) {}

  push(chunk: Buffer): void {
    if (!this.sse) {
      this.raw.push(chunk)
      return
    }
    this.buf += this.decoder.decode(chunk, { stream: true })
    let nl: number
    while ((nl = this.buf.indexOf('\n')) >= 0) {
      const line = this.buf.slice(0, nl).trim()
      this.buf = this.buf.slice(nl + 1)
      if (line.startsWith('data:')) this.event(line.slice(5).trim())
    }
  }

  private event(data: string): void {
    if (!data || data === '[DONE]') return
    let j: { choices?: { delta?: Record<string, unknown>; finish_reason?: string | null }[] }
    try {
      j = JSON.parse(data) as typeof j
    } catch {
      return
    }
    const choice = j.choices?.[0]
    if (!choice) return
    const d = choice.delta ?? {}
    if (typeof d.content === 'string') this.content += d.content
    if (typeof d.reasoning_content === 'string') this.reasoning += d.reasoning_content
    for (const t of (d.tool_calls as ToolDelta[] | undefined) ?? []) this.toolDelta(t)
    if (choice.finish_reason) this.finishReason = choice.finish_reason
  }

  private toolDelta(t: ToolDelta): void {
    const i = t.index ?? this.calls.size
    const c = this.calls.get(i) ?? { id: '', name: '', args: '' }
    if (t.id) c.id = t.id
    if (t.function?.name) c.name += t.function.name
    if (t.function?.arguments) c.args += t.function.arguments
    this.calls.set(i, c)
  }

  /** The reply, or null when the response wasn't a chat completion. */
  result(): Reply | null {
    if (!this.sse) return parseCompletion(Buffer.concat(this.raw).toString('utf8'))
    if (this.buf.trim().startsWith('data:')) this.event(this.buf.trim().slice(5).trim())
    this.buf = ''
    const toolCalls: WireToolCall[] = [...this.calls.entries()]
      .sort(([a], [b]) => a - b)
      .map(([i, c]) => ({ id: c.id || `call_${i}`, type: 'function', function: { name: c.name, arguments: c.args } }))
    if (!this.content && !this.reasoning && !toolCalls.length && !this.finishReason) return null
    return { content: this.content, reasoning: this.reasoning, toolCalls, finishReason: this.finishReason }
  }
}

function parseCompletion(text: string): Reply | null {
  let j: { choices?: { message?: { content?: unknown; reasoning_content?: unknown; tool_calls?: WireToolCall[] }; finish_reason?: string }[] }
  try {
    j = JSON.parse(text) as typeof j
  } catch {
    return null
  }
  const c = j.choices?.[0]
  if (!c?.message) return null
  const m = c.message
  return {
    content: typeof m.content === 'string' ? m.content : '',
    reasoning: typeof m.reasoning_content === 'string' ? m.reasoning_content : '',
    toolCalls: (m.tool_calls ?? []).map((t, i) => ({
      id: t.id || `call_${i}`,
      type: 'function',
      function: { name: t.function?.name ?? '', arguments: typeof t.function?.arguments === 'string' ? t.function.arguments : JSON.stringify(t.function?.arguments ?? {}) }
    })),
    finishReason: c.finish_reason ?? undefined
  }
}
