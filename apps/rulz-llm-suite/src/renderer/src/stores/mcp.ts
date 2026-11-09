import { create } from 'zustand'
import { toolTokens } from '@shared/tools'
import type { McpConfigState, McpServerStatus, McpToolInfo, ToolRequest } from '@shared/types'
import { invoke, on } from '../lib/api'
import { once } from '../lib/once'
import { useApp } from './app'

interface McpState {
  servers: McpServerStatus[]
  /** mcp.json as the main process last read it, with any read or parse error. */
  config: McpConfigState | null
  init: () => Promise<void>
}

export const useMcp = create<McpState>((set) => ({
  servers: [],
  config: null,
  init: once(async () => {
    on('mcp:changed', (servers) => set({ servers }))
    on('mcp:config', (config) => set({ config }))
    const [servers, config] = await Promise.all([invoke('mcp:status'), invoke('mcp:getConfig')])
    set({ servers, config })
  })
}))

export function findTool(qualifiedName: string): McpToolInfo | undefined {
  for (const s of useMcp.getState().servers) {
    const t = s.tools.find((x) => x.qualifiedName === qualifiedName)
    if (t) return t
  }
  return undefined
}

/** Connected tools the user has not switched off. */
export function availableTools(): McpToolInfo[] {
  const disabled = new Set(useApp.getState().settings?.disabledTools ?? [])
  return useMcp
    .getState()
    .servers.filter((s) => s.state === 'connected')
    .flatMap((s) => s.tools)
    .filter((t) => !disabled.has(t.qualifiedName))
}

export function tokensOf(names: string[]): number {
  return names.reduce((sum, n) => {
    const t = findTool(n)
    return sum + (t ? toolTokens(t) : 0)
  }, 0)
}

/** Tool options of an RE session: tools are on even when chats have them off, with the RE budget. */
export interface ToolOptions {
  re?: boolean
}

/** Token budget for loaded tool descriptions. */
export function toolBudget(opts: ToolOptions = {}): number {
  const s = useApp.getState().settings
  return (opts.re ? s?.re.toolBudgetTokens : s?.toolBudgetTokens) ?? 6000
}

/** 'all' or 'dynamic' after resolving 'auto' against the budget; null when tools are off or none exist. */
export function effectiveToolMode(opts: ToolOptions = {}): 'all' | 'dynamic' | null {
  const s = useApp.getState().settings
  if (!s || (!s.toolsEnabled && !opts.re)) return null
  const tools = availableTools()
  if (!tools.length) return null
  if (s.toolMode !== 'auto') return s.toolMode
  return tools.reduce((sum, t) => sum + toolTokens(t), 0) > toolBudget(opts) ? 'dynamic' : 'all'
}

/** The tool selection to send with a chat request. */
export function toolRequestFor(activeTools: string[] | undefined, opts: ToolOptions = {}): ToolRequest | undefined {
  const mode = effectiveToolMode(opts)
  if (!mode) return undefined
  const available = new Set(availableTools().map((t) => t.qualifiedName))
  return { mode, active: mode === 'dynamic' ? (activeTools ?? []).filter((n) => available.has(n)) : [] }
}
