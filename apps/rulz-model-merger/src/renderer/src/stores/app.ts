import { create } from 'zustand'
import { toast } from 'sonner'
import type { AppSettings, ConnectionStatus, OwnServerSettings, ServerCapabilities, ServerState, Theme } from '@shared/types'
import { call, errText, invoke } from '../lib/api'

export type Page = 'merge' | 'loras' | 'sweep' | 'results' | 'history' | 'server' | 'settings'

export function applyTheme(theme: Theme): void {
  document.documentElement.classList.toggle('dark', theme === 'dark')
}

type SettingsPatch = Omit<Partial<AppSettings>, 'ownServer'> & { ownServer?: Partial<OwnServerSettings> }

interface AppStore {
  page: Page
  settings: AppSettings | null
  serverState: ServerState | null
  status: ConnectionStatus | null
  capabilities: ServerCapabilities | null
  setPage: (p: Page) => void
  /** Saves a patch; the screen updates at once and is replaced by what the main process kept. */
  patchSettings: (patch: SettingsPatch) => Promise<void>
  setTheme: (t: Theme) => Promise<void>
  setStatus: (s: ConnectionStatus) => void
  refreshStatus: () => Promise<ConnectionStatus | undefined>
  refreshCapabilities: () => Promise<void>
  init: () => () => void
}

// Counters so a slow answer never overwrites a newer one.
let saveSeq = 0
let statusSeq = 0
let capSeq = 0
/** Changes made before the first settings:get came back; applied once it has. */
let pendingPatch: SettingsPatch | null = null

function mergePatch(a: SettingsPatch | null, b: SettingsPatch): SettingsPatch {
  return { ...a, ...b, ownServer: a?.ownServer || b.ownServer ? { ...a?.ownServer, ...b.ownServer } : undefined }
}

export const useApp = create<AppStore>((set, get) => ({
  page: 'server',
  settings: null,
  serverState: null,
  status: null,
  capabilities: null,

  setPage: (page) => set({ page }),

  patchSettings: async (patch) => {
    const cur = get().settings
    if (!cur) {
      pendingPatch = mergePatch(pendingPatch, patch)
      return
    }
    const seq = ++saveSeq
    const optimistic: AppSettings = {
      ...cur,
      ...patch,
      ownServer: { ...cur.ownServer, ...patch.ownServer }
    }
    set({ settings: optimistic })
    const saved = await call('settings:set', patch as Partial<AppSettings>)
    if (seq !== saveSeq) return
    if (saved) {
      set({ settings: saved })
      return
    }
    const fresh = await call('settings:get')
    if (fresh && seq === saveSeq) set({ settings: fresh })
  },

  setTheme: async (theme) => {
    applyTheme(theme)
    await get().patchSettings({ theme })
  },

  setStatus: (status) => {
    statusSeq++
    const was = get().status
    set({ status })
    if (!status.reachable) {
      capSeq++
      set({ capabilities: null })
    } else if (!was || !was.reachable || was.url !== status.url) {
      // A connection that just became reachable (or changed server) has new models and nodes.
      void get().refreshCapabilities()
    }
  },

  refreshStatus: async () => {
    const seq = statusSeq + 1
    statusSeq = seq
    const s = await call('comfy:status')
    if (s && seq === statusSeq) get().setStatus(s)
    return s
  },

  refreshCapabilities: async () => {
    const seq = ++capSeq
    const url = get().status?.url ?? null
    try {
      const caps = await invoke('comfy:capabilities')
      const now = get().status
      if (seq === capSeq && now?.reachable && now.url === url) set({ capabilities: caps })
    } catch (e) {
      if (seq !== capSeq) return
      set({ capabilities: null })
      toast.error(errText(e))
    }
  },

  init: () => {
    const offState = window.rulz.on('server:state', (serverState) => set({ serverState }))
    const offStatus = window.rulz.on('comfy:status', (s) => get().setStatus(s))
    void (async () => {
      const [settings, serverState] = await Promise.all([call('settings:get'), call('server:state')])
      if (serverState) set({ serverState })
      if (settings) {
        set({ settings })
        const early = pendingPatch
        pendingPatch = null
        if (early) await get().patchSettings(early)
        applyTheme(get().settings?.theme ?? settings.theme)
      }
      await get().refreshStatus()
    })()
    return () => {
      offState()
      offStatus()
    }
  }
}))
