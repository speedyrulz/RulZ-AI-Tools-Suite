import { create } from 'zustand'
import {
  availableToolsOf,
  effectiveToolModeOf,
  tokensOfNames,
  toolBudgetOf,
  toolRequestOf,
  type ToolOptions
} from '@shared/toolSession'
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

/** Connected, switched-on tools. The write/read file tools are included only for clone sessions, read_pdf never in practice sessions. */
export function availableTools(opts: { clone?: boolean; practice?: boolean } = {}): McpToolInfo[] {
  return availableToolsOf(useMcp.getState().servers, useApp.getState().settings, opts)
}

export function tokensOf(names: string[]): number {
  return tokensOfNames(useMcp.getState().servers, names)
}

export type { ToolOptions }

/** Token budget for loaded tool descriptions. */
export function toolBudget(opts: ToolOptions = {}): number {
  return toolBudgetOf(useApp.getState().settings, opts)
}

/** 'all' or 'dynamic' after resolving 'auto' against the budget; null when tools are off or none exist. */
export function effectiveToolMode(opts: ToolOptions = {}): 'all' | 'dynamic' | null {
  return effectiveToolModeOf(useMcp.getState().servers, useApp.getState().settings, opts)
}

/** The tool selection to send with a chat request. */
export function toolRequestFor(activeTools: string[] | undefined, opts: ToolOptions = {}): ToolRequest | undefined {
  return toolRequestOf(useMcp.getState().servers, useApp.getState().settings, activeTools, opts)
}
