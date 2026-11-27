import { create } from 'zustand'
import { toast } from 'sonner'
import type { GridJob, GridRequest, GridTemplateInfo } from '@shared/grid'
import { GRID_MAX_PROMPTS } from '@shared/grid'
import { call, errText, invoke } from '../lib/api'
import {
  AXIS_DEFAULT,
  activePrompts,
  clampSteps,
  dropGridImages,
  formatValues,
  generateValues,
  imageKey,
  mergeGridList,
  parseValues,
  pruneImages,
  randomSeed,
  recalledTemplatePath,
  rememberTemplatePath,
  removeGrid,
  resolveAxisOption,
  upsertGrid
} from '../lib/grid'
import { useMerge } from './merge'

interface GridStore {
  template: GridTemplateInfo | null
  loadingTemplate: boolean
  /** 'ratio', 'multiplier' or 'lora:<i>'; resolved against the recipe when used. */
  axisKey: string
  from: number
  to: number
  steps: number
  valuesText: string
  prompts: string[]
  seed: number
  grids: GridJob[]
  selectedId: string | null
  /** data: URLs of cell images, by "<gridId>/<file>". */
  images: Record<string, string>
  starting: boolean
  chooseTemplate: () => Promise<void>
  reloadRemembered: () => Promise<void>
  setAxisKey: (key: string) => void
  /** Sets the range and refills the value list from it. */
  setRange: (patch: Partial<{ from: number; to: number; steps: number }>) => void
  setValuesText: (text: string) => void
  setPrompt: (i: number, text: string) => void
  addPrompt: () => void
  removePrompt: (i: number) => void
  setSeed: (seed: number) => void
  rollSeed: () => void
  select: (id: string) => void
  loadList: () => Promise<void>
  ensureImage: (gridId: string, file: string) => Promise<void>
  /** Fetches the image now (even if another fetch is running); resolves true when it is cached afterwards. */
  fetchImageNow: (gridId: string, file: string) => Promise<boolean>
  start: () => Promise<void>
  cancel: (id: string) => Promise<void>
  remove: (id: string) => Promise<void>
  init: () => () => void
}

/** Images being fetched now, so a cell does not ask twice. */
const fetching = new Set<string>()
let templateSeq = 0

export const useGrid = create<GridStore>((set, get) => ({
  template: null,
  loadingTemplate: false,
  axisKey: 'ratio',
  from: AXIS_DEFAULT.ratio.from,
  to: AXIS_DEFAULT.ratio.to,
  steps: 5,
  valuesText: formatValues(generateValues(AXIS_DEFAULT.ratio.from, AXIS_DEFAULT.ratio.to, 5)),
  prompts: [''],
  seed: randomSeed(),
  grids: [],
  selectedId: null,
  images: {},
  starting: false,

  chooseTemplate: async () => {
    const seq = ++templateSeq
    set({ loadingTemplate: true })
    try {
      const info = await invoke('grid:loadTemplate')
      if (seq !== templateSeq) return
      if (info) applyTemplate(info)
    } catch (e) {
      if (seq === templateSeq) toast.error(errText(e))
    } finally {
      if (seq === templateSeq) set({ loadingTemplate: false })
    }
  },

  reloadRemembered: async () => {
    const path = recalledTemplatePath()
    if (!path || get().template) return
    const seq = ++templateSeq
    set({ loadingTemplate: true })
    try {
      const info = await invoke('grid:loadTemplate', path)
      if (seq !== templateSeq) return
      if (info) applyTemplate(info)
      else rememberTemplatePath(null)
    } catch {
      // The file may be gone; the user simply chooses again.
    } finally {
      if (seq === templateSeq) set({ loadingTemplate: false })
    }
  },

  setAxisKey: (axisKey) => {
    const opt = resolveAxisOption(useMerge.getState().recipe, axisKey)
    if (!opt) return
    const d = AXIS_DEFAULT[opt.axis.kind]
    const steps = get().steps
    set({ axisKey: opt.key, from: d.from, to: d.to, valuesText: formatValues(generateValues(d.from, d.to, steps)) })
  },

  setRange: (patch) => {
    const from = patch.from ?? get().from
    const to = patch.to ?? get().to
    const steps = clampSteps(patch.steps ?? get().steps)
    set({ from, to, steps, valuesText: formatValues(generateValues(from, to, steps)) })
  },

  setValuesText: (valuesText) => {
    const { values, error } = parseValues(valuesText)
    set({ valuesText, ...(error === null && values.length >= 2 ? { steps: clampSteps(values.length) } : {}) })
  },

  setPrompt: (i, text) => set({ prompts: get().prompts.map((p, j) => (j === i ? text : p)) }),
  addPrompt: () => {
    const p = get().prompts
    if (p.length < GRID_MAX_PROMPTS) set({ prompts: [...p, ''] })
  },
  removePrompt: (i) => {
    const p = get().prompts
    if (p.length > 1) set({ prompts: p.filter((_, j) => j !== i) })
  },
  setSeed: (seed) => set({ seed }),
  rollSeed: () => set({ seed: randomSeed() }),

  select: (selectedId) => set({ selectedId }),

  loadList: async () => {
    const list = await call('grid:list')
    if (!list) return
    // The list is the full state; events that arrived meanwhile replace single grids.
    const grids = mergeGridList(get().grids, list)
    set({ grids, images: pruneImages(get().images, grids) })
  },

  ensureImage: async (gridId, file) => {
    const key = imageKey(gridId, file)
    if (key in get().images || fetching.has(key)) return
    fetching.add(key)
    try {
      const url = await invoke('grid:image', gridId, file)
      if (get().grids.some((g) => g.id === gridId)) set({ images: { ...get().images, [key]: url } })
    } catch {
      // The cell shows its image as unavailable; opening the grid again retries.
    } finally {
      fetching.delete(key)
    }
  },

  fetchImageNow: async (gridId, file) => {
    const key = imageKey(gridId, file)
    if (key in get().images) return true
    try {
      const url = await invoke('grid:image', gridId, file)
      set({ images: { ...get().images, [key]: url } })
      return true
    } catch {
      return false
    }
  },

  start: async () => {
    if (get().starting) return
    const s = get()
    const recipe = useMerge.getState().recipe
    const opt = resolveAxisOption(recipe, s.axisKey)
    if (!opt || !s.template) return
    const req: GridRequest = {
      recipe,
      axis: { ...opt.axis, values: parseValues(s.valuesText).values },
      prompts: activePrompts(s.prompts),
      seed: s.seed,
      templatePath: s.template.path
    }
    set({ starting: true })
    try {
      const job = await invoke('grid:start', req)
      set({ grids: upsertGrid(get().grids, job), selectedId: job.id })
    } catch (e) {
      toast.error(errText(e))
    } finally {
      set({ starting: false })
    }
  },

  cancel: async (id) => {
    const job = await call('grid:cancel', id)
    if (job) set({ grids: upsertGrid(get().grids, job) })
  },

  remove: async (id) => {
    try {
      await invoke('grid:delete', id)
    } catch (e) {
      toast.error(errText(e))
      return
    }
    const grids = removeGrid(get().grids, id)
    set({ grids, images: dropGridImages(get().images, id), selectedId: get().selectedId === id ? null : get().selectedId })
  },

  init: () => {
    const off = window.rulz.on('grid:job', (job) => set({ grids: upsertGrid(get().grids, job) }))
    return off
  }
}))

function applyTemplate(info: GridTemplateInfo): void {
  const st = useGrid.getState()
  const prompts = st.prompts.slice()
  // The template's own prompt fills an empty first row.
  if (!info.error && info.defaultPrompt && !(prompts[0] ?? '').trim()) prompts[0] = info.defaultPrompt
  useGrid.setState({ template: info, prompts })
  if (!info.error) rememberTemplatePath(info.path)
}
