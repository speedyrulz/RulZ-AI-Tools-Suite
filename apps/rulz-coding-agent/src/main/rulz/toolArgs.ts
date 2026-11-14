import type { WireMessage, WireToolCall } from '@shared/types'
import { parseToolArgs } from '../agent/tools'

/**
 * Both fixes below run over every tool call in the history before every turn. A call's result
 * depends only on its arguments text, and the history keeps the same call objects from turn to
 * turn, so each call is worked out once and remembered (per call object, so nothing is kept alive
 * after the history drops the call). The arguments text is checked too, in case it was changed.
 */
type CallCache = WeakMap<WireToolCall, { raw: string; out: WireToolCall }>

function cachedCall(cache: CallCache, c: WireToolCall, work: (raw: string) => WireToolCall): WireToolCall {
  const raw = c.function.arguments ?? ''
  const hit = cache.get(c)
  if (hit && hit.raw === raw) return hit.out
  const out = work(raw)
  cache.set(c, { raw, out })
  return out
}

const validCache: CallCache = new WeakMap()
const jsonCache: CallCache = new WeakMap()

/**
 * Makes every earlier tool call's arguments a JSON object. llama.cpp parses them to fill in the
 * chat template and rejects the whole request when one doesn't parse, so a single call the model
 * cut off mid-string (a reply that ran out of tokens or was interrupted) would fail every later
 * turn of the task. Python-style arguments are converted; arguments that can't be read become {}
 * (the tool's result already told the model they were invalid).
 */
export function validToolCallArgs(messages: WireMessage[]): WireMessage[] {
  let changed = false
  const out = messages.map((m) => {
    if (!m.tool_calls?.length) return m
    let fixed = false
    const calls = m.tool_calls.map((c) => {
      const v = cachedCall(validCache, c, (raw) => validCall(c, raw))
      if (v !== c) fixed = true
      return v
    })
    if (!fixed) return m
    changed = true
    return { ...m, tool_calls: calls }
  })
  return changed ? out : messages
}

function validCall(c: WireToolCall, raw: string): WireToolCall {
  try {
    const v: unknown = JSON.parse(raw)
    if (v && typeof v === 'object' && !Array.isArray(v)) return c
  } catch {
    // repaired below
  }
  const parsed = parseToolArgs(raw)
  return { ...c, function: { ...c.function, arguments: 'args' in parsed ? JSON.stringify(parsed.args) : '{}' } }
}

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
    return { ...m, tool_calls: m.tool_calls.map((c) => cachedCall(jsonCache, c, (raw) => jsonifyCall(c, raw))) }
  })
}

function jsonifyCall(c: WireToolCall, raw: string): WireToolCall {
  let args: unknown
  try {
    args = JSON.parse(raw || '{}')
  } catch {
    return c
  }
  if (!args || typeof args !== 'object' || Array.isArray(args)) return c
  const out: Record<string, string> = {}
  for (const [k, v] of Object.entries(args)) out[k] = typeof v === 'string' ? v : JSON.stringify(v)
  return { ...c, function: { ...c.function, arguments: JSON.stringify(out) } }
}
