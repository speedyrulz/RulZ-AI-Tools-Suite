import { create } from 'zustand'
import { toast } from 'sonner'
import type { GridJob, GridRequest, GridRowKind, GridTemplateInfo } from '@shared/grid'
import { GRID_MAX_PROMPTS } from '@shared/grid'
import { call, errText, invoke } from '../lib/api'
import {
  AXIS_DEFAULT,
  addSeed,
  buildGridRequest,
  clampSteps,
  dropGridImages,
  formFromRequest,
  formatValues,
  generateValues,
  imageKey,
  mergeGridList,
  parseValues,
  pruneImages,
  randomSeed,
  randomSeeds,
  recalledTemplatePath,
  rememberTemplatePath,
  removeGrid,
  removeSeed,
  resolveAxisOption,
  setSeedAt,
  upsertGrid
} from '../lib/grid'
import { mergePageWillChange } from '../lib/history'
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
  /** What the rows are. Seed rows use the first prompt only. */
  rowKind: GridRowKind
  prompts: string[]
  seed: number
  /** Seed rows: 1..GRID_MAX_SEEDS distinct seeds. */
  seeds: number[]
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
  setRowKind: (kind: GridRowKind) => void
  setSeedAt: (i: number, seed: number) => void
  addSeedRow: () => void
  removeSeedRow: (i: number) => void
  /** Gives every seed row a new random seed (all different). */
  rollSeeds: () => void
  /** Fills the Sweep form from a finished grid's request (Reopen): the recipe goes to the Merge store, the rest to this
   *  store, and the workflow is loaded again from its path. Resolves true when the workflow loaded without error; on
   *  failure a toast says so, the workflow is cleared and the rest stays filled. */
  loadRequest: (req: GridRequest) => Promise<boolean>
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

/** Grids whose images are kept besides the selected one; the rest are dropped (each image is a full PNG as base64). */
const MAX_CACHED_GRIDS = 3
/** Grid ids, most recently used (an image was added for it) first. */
const recentGrids: string[] = []

/** The cache with one more image; the grid it belongs to becomes the most recently used and the images of grids
 *  beyond the newest MAX_CACHED_GRIDS (the selected grid excepted) are dropped. Shown grids reload what they miss. */
/** Marks a grid as just used (shown, selected or its cached images read), without dropping anything yet. */
function touchGrid(gridId: string): void {
  const at = recentGrids.indexOf(gridId)
  if (at === 0) return
  if (at > 0) recentGrids.splice(at, 1)
  recentGrids.unshift(gridId)
  recentGrids.length = Math.min(recentGrids.length, MAX_CACHED_GRIDS)
}

function addImage(images: Record<string, string>, selectedId: string | null, gridId: string, key: string, url: string): Record<string, string> {
  touchGrid(gridId)
  const keep = new Set(recentGrids)
  if (selectedId) keep.add(selectedId)
  const next: Record<string, string> = {}
  for (const [k, v] of Object.entries(images)) if (keep.has(k.slice(0, k.indexOf('/')))) next[k] = v
  next[key] = url
  return next
}

export const useGrid = create<GridStore>((set, get) => ({
  template: null,
  loadingTemplate: false,
  axisKey: 'ratio',
  from: AXIS_DEFAULT.ratio.from,
  to: AXIS_DEFAULT.ratio.to,
  steps: 5,
  valuesText: formatValues(generateValues(AXIS_DEFAULT.ratio.from, AXIS_DEFAULT.ratio.to, 5)),
  rowKind: 'prompts',
  prompts: [''],
  seed: randomSeed(),
  seeds: randomSeeds(2),
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

  setRowKind: (rowKind) => {
    if (rowKind === get().rowKind) return
    set({ rowKind })
    if (get().seeds.length === 0) set({ seeds: randomSeeds(2) })
  },
  setSeedAt: (i, seed) => set({ seeds: setSeedAt(get().seeds, i, seed) }),
  addSeedRow: () => set({ seeds: addSeed(get().seeds) }),
  removeSeedRow: (i) => set({ seeds: removeSeed(get().seeds, i) }),
  rollSeeds: () => set({ seeds: randomSeeds(Math.max(1, get().seeds.length)) }),

  loadRequest: async (req) => {
    const f = formFromRequest(req)
    // As "Use this value" does: the recipe goes to the Merge page, which keeps its own output name.
    const current = useMerge.getState().recipe
    if (mergePageWillChange(current, req.recipe)) toast.info("The Merge page now holds this grid's merge.")
    useMerge.getState().replaceRecipe({ ...req.recipe, outputName: current.outputName })
    set({
      axisKey: f.axisKey,
      from: f.from,
      to: f.to,
      steps: f.steps,
      valuesText: f.valuesText,
      rowKind: f.rowKind,
      prompts: f.prompts,
      seed: f.seed,
      ...(f.seeds ? { seeds: f.seeds } : {})
    })
    const seq = ++templateSeq
    set({ loadingTemplate: true })
    try {
      const info = await invoke('grid:loadTemplate', req.templatePath)
      if (seq !== templateSeq) return false
      if (!info) throw new Error('The file was not found.')
      set({ template: info })
      if (info.error) {
        toast.error(`The workflow ${info.name} has a problem: ${info.error}`)
        return false
      }
      rememberTemplatePath(info.path)
      return true
    } catch (e) {
      if (seq !== templateSeq) return false
      set({ template: null })
      toast.error(`The workflow of this grid could not be loaded (${req.templatePath}): ${errText(e)} Choose a workflow again.`)
      return false
    } finally {
      if (seq === templateSeq) set({ loadingTemplate: false })
    }
  },

  select: (selectedId) => {
    touchGrid(selectedId)
    set({ selectedId })
  },

  loadList: async () => {
    const list = await call('grid:list')
    if (!list) return
    // The list is the full state; events that arrived meanwhile replace single grids.
    const grids = mergeGridList(get().grids, list)
    set({ grids, images: pruneImages(get().images, grids) })
  },

  ensureImage: async (gridId, file) => {
    const key = imageKey(gridId, file)
    if (key in get().images) {
      touchGrid(gridId)
      return
    }
    if (fetching.has(key)) return
    fetching.add(key)
    try {
      const url = await invoke('grid:image', gridId, file)
      if (get().grids.some((g) => g.id === gridId)) set({ images: addImage(get().images, get().selectedId, gridId, key, url) })
    } catch {
      // The cell shows its image as unavailable; opening the grid again retries.
    } finally {
      fetching.delete(key)
    }
  },

  fetchImageNow: async (gridId, file) => {
    const key = imageKey(gridId, file)
    if (key in get().images) {
      touchGrid(gridId)
      return true
    }
    try {
      const url = await invoke('grid:image', gridId, file)
      set({ images: addImage(get().images, get().selectedId, gridId, key, url) })
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
    const req: GridRequest = buildGridRequest({
      recipe,
      axis: opt,
      valuesText: s.valuesText,
      rowKind: s.rowKind,
      prompts: s.prompts,
      seeds: s.seeds,
      seed: s.seed,
      templatePath: s.template.path
    })
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
