import { create } from 'zustand'
import { toast } from 'sonner'
import type { LoraMergeCheck, LoraMergeDtype, LoraMergeJob, LoraMergeMode } from '@shared/loraMerge'
import { call, errText, invoke } from '../lib/api'
import { buildInspectRequest, buildRecipe, emptyRows, namedList, suggestLoraOutputName, upsertLoraJob, type LoraRow } from '../lib/loraMerge'

interface LoraMergeStore {
  rows: LoraRow[]
  mode: LoraMergeMode
  rank: number
  dtype: LoraMergeDtype
  outputName: string
  /** True once the user typed an output name; until then it follows the named inputs. */
  nameTouched: boolean
  check: LoraMergeCheck | null
  checking: boolean
  checkError: string | null
  jobs: LoraMergeJob[]
  starting: boolean
  list: string[]
  listLoading: boolean
  listError: string | null
  patchRows: (fn: (rows: LoraRow[]) => LoraRow[]) => void
  setOptions: (patch: Partial<{ mode: LoraMergeMode; rank: number; dtype: LoraMergeDtype }>) => void
  setOutputName: (name: string) => void
  runCheck: () => void
  loadList: () => Promise<void>
  start: () => Promise<void>
  cancel: (id: string) => Promise<void>
  init: () => () => void
}

let checkSeq = 0
let checkTimer: number | undefined
let listSeq = 0

export const useLoraMerge = create<LoraMergeStore>((set, get) => ({
  rows: emptyRows(),
  mode: 'concat',
  rank: 64,
  dtype: 'source',
  outputName: 'merged-lora',
  nameTouched: false,
  check: null,
  checking: false,
  checkError: null,
  jobs: [],
  starting: false,
  list: [],
  listLoading: false,
  listError: null,

  patchRows: (fn) => {
    const before = get().rows
    const rows = fn(before)
    if (rows === before) return
    const namesChanged = namedList(before).join('\n') !== namedList(rows).join('\n')
    const patch: Partial<LoraMergeStore> = { rows }
    if (!get().nameTouched && namesChanged) patch.outputName = suggestLoraOutputName(rows)
    set(patch)
    // Only the named inputs (and their order) make a new question; weights do not.
    if (namesChanged) {
      set({ check: null, checkError: null })
      get().runCheck()
    }
  },

  setOptions: (patch) => {
    set(patch)
    set({ check: null, checkError: null })
    get().runCheck()
  },

  setOutputName: (outputName) => set({ outputName, nameTouched: outputName.trim() !== '' }),

  runCheck: () => {
    window.clearTimeout(checkTimer)
    const seq = ++checkSeq
    const { rows, mode, rank, dtype } = get()
    if (namedList(rows).length < 2) {
      set({ check: null, checking: false, checkError: null })
      return
    }
    set({ checking: true, checkError: null })
    checkTimer = window.setTimeout(() => {
      void (async () => {
        try {
          const res = await invoke('loraMerge:inspect', buildInspectRequest(rows, { mode, rank, dtype }))
          if (seq !== checkSeq) return
          set({ check: res, checking: false, checkError: null })
        } catch (e) {
          if (seq !== checkSeq) return
          set({ check: null, checking: false, checkError: errText(e) })
        }
      })()
    }, 400)
  },

  loadList: async () => {
    const seq = ++listSeq
    set({ listLoading: true, listError: null })
    try {
      const list = await invoke('loraMerge:list')
      if (seq !== listSeq) return
      set({ list, listLoading: false })
    } catch (e) {
      if (seq !== listSeq) return
      set({ list: [], listLoading: false, listError: errText(e) })
    }
  },

  start: async () => {
    if (get().starting) return
    set({ starting: true })
    try {
      const { rows, mode, rank, dtype, outputName } = get()
      const job = await invoke('loraMerge:start', buildRecipe(rows, { mode, rank, dtype, outputName }))
      set({ jobs: upsertLoraJob(get().jobs, job) })
    } catch (e) {
      toast.error(errText(e))
    } finally {
      set({ starting: false })
    }
  },

  cancel: async (id) => {
    const job = await call('loraMerge:cancel', id)
    if (job) set({ jobs: upsertLoraJob(get().jobs, job) })
  },

  init: () => {
    const off = window.rulz.on('loraMerge:job', (job) => set({ jobs: upsertLoraJob(get().jobs, job) }))
    void (async () => {
      const jobs = await call('loraMerge:jobs')
      if (!jobs) return
      let next = get().jobs
      for (const j of jobs) next = upsertLoraJob(next, j)
      set({ jobs: next })
    })()
    return off
  }
}))
