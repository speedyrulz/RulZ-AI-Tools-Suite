import { create } from 'zustand'
import type { McpServerStatus, McpToolInfo } from '@shared/types'
import { invoke, on } from '../lib/api'

interface McpState {
  servers: McpServerStatus[]
  init: () => Promise<void>
}

export const useMcp = create<McpState>((set) => ({
  servers: [],
  init: async () => {
    on('mcp:changed', (servers) => set({ servers }))
    set({ servers: await invoke('mcp:status') })
  }
}))

export function findTool(qualifiedName: string): McpToolInfo | undefined {
  for (const s of useMcp.getState().servers) {
    const t = s.tools.find((x) => x.qualifiedName === qualifiedName)
    if (t) return t
  }
  return undefined
}
