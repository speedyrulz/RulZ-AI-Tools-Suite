import type { WireMessage } from '@shared/types'

/**
 * True for chat templates that write each tool-call argument as an XML-style `<parameter=…>` block
 * and turn non-string values into text with Jinja's `| string` (Qwen3-Coder's original template,
 * for example). That prints lists and objects in Python form (`['a', 'b']`, `True`), the model
 * copies that form in its next call, and llama.cpp can't read it as JSON, so the values are lost.
 */
export function templateStringifiesArgs(template: string | undefined): boolean {
  if (!template || !template.includes('<parameter=')) return false
  return /args_value\s*\|\s*string/.test(template) && !/args_value\s*\|\s*tojson/.test(template)
}

/**
 * Rewrites earlier tool calls so every non-string argument value is already JSON text. Templates
 * that print values as-is then show the model valid JSON (`["a", "b"]`) to imitate.
 */
export function jsonifyToolCallArgs(messages: WireMessage[]): WireMessage[] {
  return messages.map((m) => {
    if (!m.tool_calls?.length) return m
    return {
      ...m,
      tool_calls: m.tool_calls.map((c) => {
        let args: unknown
        try {
          args = JSON.parse(c.function.arguments || '{}')
        } catch {
          return c
        }
        if (!args || typeof args !== 'object' || Array.isArray(args)) return c
        const out: Record<string, string> = {}
        for (const [k, v] of Object.entries(args)) out[k] = typeof v === 'string' ? v : JSON.stringify(v)
        return { ...c, function: { ...c.function, arguments: JSON.stringify(out) } }
      })
    }
  })
}
