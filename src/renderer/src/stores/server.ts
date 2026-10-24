import { create } from 'zustand'
import { toast } from 'sonner'
import type { ApiRequestLog, ApiServerStatus } from '@shared/types'
import { errorMessage, invoke, on } from '../lib/api'

const MAX_LOG = 300

interface ServerState {
  status: ApiServerStatus | null
  log: ApiRequestLog[]
  busy: boolean
  init: () => Promise<void>
  start: () => Promise<void>
  stop: () => Promise<void>
  clearLog: () => Promise<void>
}

export const useServer = create<ServerState>((set, get) => ({
  status: null,
  log: [],
  busy: false,
  init: async () => {
    on('server:changed', (status) => set({ status }))
    on('server:request', (entry) => {
      const log = get().log
      const i = log.findIndex((e) => e.id === entry.id)
      const next = i >= 0 ? log.map((e) => (e.id === entry.id ? entry : e)) : [...log, entry].slice(-MAX_LOG)
      set({ log: next })
    })
    const [status, log] = await Promise.all([invoke('server:status'), invoke('server:log')])
    set({ status, log })
  },
  start: async () => {
    set({ busy: true })
    try {
      set({ status: await invoke('server:start') })
    } catch (e) {
      toast.error(errorMessage(e))
    } finally {
      set({ busy: false })
    }
  },
  stop: async () => {
    set({ busy: true })
    try {
      set({ status: await invoke('server:stop') })
    } finally {
      set({ busy: false })
    }
  },
  clearLog: async () => {
    await invoke('server:clearLog')
    set({ log: [] })
  }
}))
