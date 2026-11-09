import { create } from 'zustand'
import { sameComputer, type Computer } from '../lib/computers'
import { deviceName, loadPref, savePref } from '../lib/native'
import { link, type LinkState } from '../lib/remote'

interface LinkStore {
  loaded: boolean
  computers: Computer[]
  currentId: string | null
  state: LinkState
  /** What this phone is called on the computer. */
  device: string
  load: () => Promise<void>
  /** Saves a paired computer (replacing the same one paired before) and connects to it. */
  add: (computer: Omit<Computer, 'id'>) => Promise<void>
  remove: (id: string) => Promise<void>
  /** Connects to a saved computer, or with null disconnects (back to the list of computers). */
  select: (id: string | null) => void
  setDevice: (name: string) => Promise<void>
}

export const useLink = create<LinkStore>((set, get) => {
  link.onState((state) => set({ state }))

  const persist = async (): Promise<void> => {
    await savePref('computers', get().computers)
    await savePref('currentComputer', get().currentId)
  }

  return {
    loaded: false,
    computers: [],
    currentId: null,
    state: link.state,
    device: '',

    load: async () => {
      const [computers, currentId, saved] = await Promise.all([
        loadPref<Computer[]>('computers', []),
        loadPref<string | null>('currentComputer', null),
        loadPref<string>('deviceName', '')
      ])
      const device = saved || (await deviceName())
      set({ computers, currentId, device, loaded: true })
      const current = computers.find((c) => c.id === currentId)
      if (current) link.connect(current, device)
    },

    add: async (computer) => {
      const existing = get().computers.find((c) => sameComputer(c, computer))
      const saved: Computer = { ...computer, id: existing?.id ?? crypto.randomUUID() }
      set((s) => ({ computers: [...s.computers.filter((c) => c.id !== saved.id), saved], currentId: saved.id }))
      await persist()
      link.connect(saved, get().device)
    },

    remove: async (id) => {
      if (get().currentId === id) link.disconnect()
      set((s) => ({ computers: s.computers.filter((c) => c.id !== id), currentId: s.currentId === id ? null : s.currentId }))
      await persist()
    },

    select: (id) => {
      const c = get().computers.find((x) => x.id === id)
      set({ currentId: c?.id ?? null })
      void persist()
      if (c) link.connect(c, get().device)
      else link.disconnect()
    },

    setDevice: async (name) => {
      const device = name.trim() || (await deviceName())
      set({ device })
      await savePref('deviceName', device)
    }
  }
})

export function currentComputer(): Computer | undefined {
  const { computers, currentId } = useLink.getState()
  return computers.find((c) => c.id === currentId)
}
