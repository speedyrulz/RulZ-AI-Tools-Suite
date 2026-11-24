// Which tools a chat or an unattended session offers, and what the built-in loader tools do. Pure
// functions over the connected servers and a few settings, so the chat store (window) and the
// practice sessions (main process) behave the same.
import { CLONE_FILE_TOOLS, PRACTICE_HIDDEN_TOOLS } from './re'
import { META_LOAD, META_SEARCH, resolveTool, searchTools, toolTokens } from './tools'
import type { FileToolScope, McpServerStatus, McpToolInfo, ToolCall, ToolMode, ToolRequest, ToolResult } from './types'

/** The settings the tool choice reads. */
export interface ToolSettings {
  toolsEnabled: boolean
  toolMode: ToolMode
  toolBudgetTokens: number
  disabledTools: string[]
  re: { toolBudgetTokens: number }
}

/** Tool options of a session: RE sessions have tools on even when chats have them off, with the RE budget. */
export interface ToolOptions {
  re?: boolean
  /** Clone sessions: where the write/read file tools may work. */
  files?: FileToolScope
  /** Practice sessions (attended or batch): no read_pdf. */
  practice?: boolean
}

/** Connected, switched-on tools. The write/read file tools are included only for clone sessions, read_pdf never in practice sessions. */
export function availableToolsOf(servers: McpServerStatus[], settings: Pick<ToolSettings, 'disabledTools'> | null | undefined, opts: { clone?: boolean; practice?: boolean } = {}): McpToolInfo[] {
  const disabled = new Set(settings?.disabledTools ?? [])
  if (!opts.clone) for (const n of CLONE_FILE_TOOLS) disabled.add(n)
  if (opts.practice) for (const n of PRACTICE_HIDDEN_TOOLS) disabled.add(n)
  return servers
    .filter((s) => s.state === 'connected')
    .flatMap((s) => s.tools)
    .filter((t) => !disabled.has(t.qualifiedName))
}

/** Tokens the named tools take (looked up in every server's tools, connected or not). */
export function tokensOfNames(servers: McpServerStatus[], names: string[]): number {
  return names.reduce((sum, n) => {
    for (const s of servers) {
      const t = s.tools.find((x) => x.qualifiedName === n)
      if (t) return sum + toolTokens(t)
    }
    return sum
  }, 0)
}

/** Token budget for loaded tool descriptions. */
export function toolBudgetOf(settings: Pick<ToolSettings, 'toolBudgetTokens' | 're'> | null | undefined, opts: ToolOptions = {}): number {
  return (opts.re ? settings?.re.toolBudgetTokens : settings?.toolBudgetTokens) ?? 6000
}

/** 'all' or 'dynamic' after resolving 'auto' against the budget; null when tools are off or none exist. */
export function effectiveToolModeOf(servers: McpServerStatus[], settings: ToolSettings | null | undefined, opts: ToolOptions = {}): 'all' | 'dynamic' | null {
  const s = settings
  if (!s || (!s.toolsEnabled && !opts.re)) return null
  const tools = availableToolsOf(servers, s, { clone: !!opts.files, practice: opts.practice })
  if (!tools.length) return null
  if (s.toolMode !== 'auto') return s.toolMode
  return tools.reduce((sum, t) => sum + toolTokens(t), 0) > toolBudgetOf(s, opts) ? 'dynamic' : 'all'
}

/** The tool selection to send with a chat request. */
export function toolRequestOf(
  servers: McpServerStatus[],
  settings: ToolSettings | null | undefined,
  activeTools: string[] | undefined,
  opts: ToolOptions = {}
): ToolRequest | undefined {
  const mode = effectiveToolModeOf(servers, settings, opts)
  if (!mode) return undefined
  const available = new Set(availableToolsOf(servers, settings, { clone: !!opts.files, practice: opts.practice }).map((t) => t.qualifiedName))
  return {
    mode,
    active: mode === 'dynamic' ? (activeTools ?? []).filter((n) => available.has(n)) : [],
    ...(opts.files ? { files: opts.files } : {}),
    ...(opts.practice ? { practice: true } : {})
  }
}

/**
 * Marks tools as loaded and most recently used (last in the list), then evicts the least recently used
 * while the list is over `budget`. Never evicts the tools just loaded.
 */
export function loadToolNames(active: string[], names: string[], budget: number, tokensOf: (names: string[]) => number): { active: string[]; evicted: string[] } {
  const next = [...active.filter((n) => !names.includes(n)), ...names]
  const evicted: string[] = []
  while (next.length > names.length && tokensOf(next) > budget) evicted.push(next.shift()!)
  return { active: next, evicted }
}

/** Tool names from a list, or from text such as "a, b" or "['a', 'b']" that some models send. */
export function toolNameList(value: unknown): string[] {
  const raw = Array.isArray(value) ? value.map(String) : typeof value === 'string' ? value.split(/[\s,;]+/) : []
  return raw.map((n) => n.replace(/^[\s[\]'"`]+|[\s[\]'"`]+$/g, '')).filter(Boolean)
}

/**
 * Runs one of the built-in loader tools (search, load, unload). They change the loaded-tools list only, so they
 * need no approval. Returns the result and the loaded-tools list as it is afterwards.
 */
export function runLoaderTool(
  call: Pick<ToolCall, 'name' | 'arguments'>,
  tools: McpToolInfo[],
  active: string[],
  budget: number,
  tokensOf: (names: string[]) => number
): { result: ToolResult; active: string[] } {
  let args: { query?: string; limit?: number; names?: unknown } = {}
  try {
    const parsed: unknown = call.arguments.trim() ? JSON.parse(call.arguments) : {}
    // `null` (or a bare number) parses too; treat it as no arguments rather than throw on it.
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) args = parsed
  } catch {
    return { result: { text: `The arguments were not valid JSON: ${call.arguments}`, isError: true }, active }
  }
  const loaded = new Set(active)
  if (call.name === META_SEARCH) {
    const hits = searchTools(tools, String(args.query ?? ''), Number(args.limit) || 10)
    if (!hits.length) return { result: { text: `No tools match "${args.query ?? ''}".`, isError: false }, active }
    return {
      result: {
        text: hits
          .map((h) => `- ${h.name} (${h.server})${loaded.has(resolveTool(tools, `${h.server}__${h.name}`)?.qualifiedName ?? '') ? ' [loaded]' : ''}: ${h.summary}`)
          .join('\n'),
        isError: false
      },
      active
    }
  }
  const names = toolNameList(args.names)
  if (!names.length) {
    return {
      result: {
        text: 'No tool names arrived: "names" was empty. Pass a JSON list with double quotes, for example ["analyze_javascript_application"].',
        isError: true
      },
      active
    }
  }
  const resolved = names.map((n) => ({ n, t: resolveTool(tools, n) }))
  const unknown = resolved
    .filter((r) => !r.t)
    .map((r) => {
      const guess = searchTools(tools, r.n, 3).map((h) => h.name)
      return guess.length ? `${r.n} (did you mean ${guess.join(', ')}?)` : r.n
    })
  const found = [...new Set(resolved.flatMap((r) => (r.t ? [r.t.qualifiedName] : [])))]
  const lines: string[] = []
  let next = active
  if (call.name === META_LOAD) {
    if (found.length) {
      const r = loadToolNames(active, found, budget, tokensOf)
      next = r.active
      lines.push(`Loaded: ${found.join(', ')}. You can call them in your next step.`)
      if (r.evicted.length) lines.push(`Unloaded to stay within the tool budget (least recently used): ${r.evicted.join(', ')}.`)
    }
  } else if (found.length) {
    next = active.filter((n) => !found.includes(n))
    lines.push(`Unloaded: ${found.join(', ')}.`)
  }
  if (unknown.length) lines.push(`Unknown tools: ${unknown.join('; ')}.`)
  return { result: { text: lines.join('\n'), isError: found.length === 0 }, active: next }
}
