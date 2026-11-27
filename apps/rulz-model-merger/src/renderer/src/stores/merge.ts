import { create } from 'zustand'
import { toast } from 'sonner'
import { DEFAULT_RECIPE, type LoraEntry, type MergeCheck, type MergeJob, type MergeRecipe } from '@shared/merge'
import { call, errText, invoke } from '../lib/api'
import { loraNames, suggestOutputName, upsertJob } from '../lib/merge'

interface MergeStore {
  recipe: MergeRecipe
  check: MergeCheck | null
  checking: boolean
  /** Text of the last failed inspect call; null when the last call worked or none is pending. */
  checkError: string | null
  jobs: MergeJob[]
  starting: boolean
  /** True once the user typed an output name; until then the name follows A and the LoRAs. */
  nameTouched: boolean
  /** Changes the recipe; a change to kind, mode, A, B or C re-runs the check after a short pause. */
  setRecipe: (patch: Partial<MergeRecipe>) => void
  /** Replaces the LoRA list with the result of `fn` (immutable helpers in lib/merge). */
  patchLoras: (fn: (loras: LoraEntry[]) => LoraEntry[]) => void
  setBlock: (inputs: string[], value: number) => void
  resetBlocks: () => void
  runCheck: () => void
  start: () => Promise<void>
  cancel: (id: string) => Promise<void>
  moveToModels: (id: string) => Promise<void>
  init: () => () => void
}

let checkSeq = 0
let checkTimer: number | undefined
const CHECK_KEYS: (keyof MergeRecipe)[] = ['kind', 'mode', 'a', 'b', 'c']
const sameLoras = (x: MergeRecipe, y: MergeRecipe): boolean => loraNames(x.loras).join('|') === loraNames(y.loras).join('|')

export const useMerge = create<MergeStore>((set, get) => ({
  recipe: { ...DEFAULT_RECIPE },
  check: null,
  checking: false,
  checkError: null,
  jobs: [],
  starting: false,
  nameTouched: false,

  setRecipe: (patch) => {
    const before = get().recipe
    let next: MergeRecipe = { ...before, ...patch }
    // Another kind has other file lists, and another block node has other inputs.
    if (patch.kind && patch.kind !== before.kind) {
      next = {
        ...next,
        a: '',
        b: '',
        c: null,
        blockNode: null,
        blocks: {},
        weightDtype: 'default'
      }
    }
    if (patch.blockNode !== undefined && patch.blockNode !== before.blockNode && patch.blocks === undefined) {
      next = { ...next, blocks: {} }
    }
    let touched = get().nameTouched
    if (patch.outputName !== undefined) touched = patch.outputName.trim() !== ''
    else if (!touched && (before.a !== next.a || !sameLoras(before, next) || patch.kind)) next = { ...next, outputName: suggestOutputName(next) }
    set({ recipe: next, nameTouched: touched })
    if (CHECK_KEYS.some((k) => before[k] !== next[k]) || !sameLoras(before, next)) {
      // A new question makes the shown answer stale.
      set({ check: null, checkError: null })
      get().runCheck()
    }
  },

  patchLoras: (fn) => {
    const loras = fn(get().recipe.loras)
    if (loras !== get().recipe.loras) get().setRecipe({ loras })
  },

  setBlock: (inputs, value) => {
    const blocks = { ...get().recipe.blocks }
    for (const i of inputs) blocks[i] = value
    set({ recipe: { ...get().recipe, blocks } })
  },

  resetBlocks: () => set({ recipe: { ...get().recipe, blocks: {} } }),

  runCheck: () => {
    window.clearTimeout(checkTimer)
    const seq = ++checkSeq
    const r = get().recipe
    if (!r.a || (r.mode !== 'bake' && !r.b) || (r.mode === 'addDifference' && !r.c)) {
      set({ check: null, checking: false, checkError: null })
      return
    }
    set({ checking: true, checkError: null })
    checkTimer = window.setTimeout(() => {
      void (async () => {
        try {
          const res = await invoke('merge:inspect', {
            kind: r.kind,
            mode: r.mode,
            a: r.a,
            b: r.b,
            c: r.mode === 'addDifference' ? r.c : null,
            loras: loraNames(r.loras)
          })
          if (seq !== checkSeq) return
          const cur = get().recipe
          let blockNode = cur.blockNode
          if (cur.mode === 'blocks' && (!blockNode || !res.blockNodes.some((b) => b.node === blockNode))) {
            blockNode = res.suggestedBlockNode ?? res.blockNodes[0]?.node ?? null
          }
          set({
            check: res,
            checking: false,
            checkError: null,
            recipe: blockNode === cur.blockNode ? cur : { ...cur, blockNode, blocks: {} }
          })
        } catch (e) {
          if (seq !== checkSeq) return
          set({ check: null, checking: false, checkError: errText(e) })
        }
      })()
    }, 400)
  },

  start: async () => {
    if (get().starting) return
    set({ starting: true })
    try {
      const job = await invoke('merge:start', get().recipe)
      set({ jobs: upsertJob(get().jobs, job) })
    } catch (e) {
      toast.error(errText(e))
    } finally {
      set({ starting: false })
    }
  },

  cancel: async (id) => {
    const job = await call('merge:cancel', id)
    if (job) set({ jobs: upsertJob(get().jobs, job) })
  },

  moveToModels: async (id) => {
    const job = await call('merge:moveToModels', id)
    if (job) set({ jobs: upsertJob(get().jobs, job) })
  },

  init: () => {
    const off = window.rulz.on('merge:job', (job) => set({ jobs: upsertJob(get().jobs, job) }))
    void (async () => {
      const jobs = await call('merge:jobs')
      if (!jobs) return
      // The list is the full state; events that arrive later replace single jobs.
      let next = get().jobs
      for (const j of jobs) next = upsertJob(next, j)
      set({ jobs: next })
    })()
    return off
  }
}))
