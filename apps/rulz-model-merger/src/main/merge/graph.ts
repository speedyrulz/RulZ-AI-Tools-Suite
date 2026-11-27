// Builds the ComfyUI API-format prompt for a merge recipe. Pure; ids are stable strings so output is deterministic.
import type { MergeRecipe } from '@shared/merge'
import type { ServerCapabilities } from '@shared/types'

export interface BuiltGraph {
  prompt: Record<string, unknown>
  saveNode: 'ModelSave' | 'CheckpointSave'
  outputSubfolder: 'diffusion_models' | 'checkpoints'
  filenamePrefix: string
}

type Link = [string, number]

const r4 = (n: number): number => Math.round(n * 10000) / 10000
const clamp = (n: number, lo: number, hi: number): number => (Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : lo)

/** Keep letters, digits, space - _ . ; runs of anything else become one _. No leading dots, 120 chars at most. */
export function sanitizeOutputName(s: string): string {
  let out = (s ?? '').replace(/[^A-Za-z0-9 _.-]+/g, '_').trim()
  out = out.replace(/^\.+/, '').slice(0, 120).trim()
  return out === '' ? 'merged' : out
}

/** Problems that stop a run, as plain sentences. `blockInputs` = the chosen block node's FLOAT inputs, when known. */
export function validateRecipe(recipe: MergeRecipe, caps: ServerCapabilities, blockInputs: string[] | null): string[] {
  const problems: string[] = []
  const isCkpt = recipe.kind === 'checkpoint'
  const list = isCkpt ? caps.checkpoints : caps.diffusionModels
  const listName = isCkpt ? 'checkpoints' : 'diffusion models'
  const need = (node: string): void => {
    if (!caps.nodes[node]) problems.push(`The server has no ${node} node.`)
  }
  const checkName = (label: string, name: string | null): void => {
    if (!name) problems.push(`Choose model ${label}.`)
    else if (!list.includes(name)) problems.push(`Model ${label} "${name}" is not in the server's ${listName} list.`)
  }
  checkName('A', recipe.a)
  checkName('B', recipe.b)
  if (recipe.a && recipe.a === recipe.b) problems.push('A and B are the same file. Choose two different models.')
  if (recipe.mode === 'addDifference') {
    checkName('C', recipe.c)
    if (recipe.c && recipe.c === recipe.a) problems.push('C is the same file as A. Choose a different model C.')
  }
  need(isCkpt ? 'CheckpointLoaderSimple' : 'UNETLoader')
  need(isCkpt ? 'CheckpointSave' : 'ModelSave')
  if (recipe.mode === 'weighted') need('ModelMergeSimple')
  if (recipe.mode === 'addDifference') {
    need('ModelMergeSubtract')
    need('ModelMergeAdd')
  }
  if (isCkpt && recipe.clip === 'merge' && recipe.mode !== 'addDifference') need('CLIPMergeSimple')
  // CLIPMergeSubtract / CLIPMergeAdd are not in caps.nodes: accepted here, the server reports them if missing.

  if (recipe.mode === 'blocks') {
    if (!recipe.blockNode) problems.push('Choose a block merge node.')
    else if (recipe.blockNode === 'ModelMergeBlocks') need('ModelMergeBlocks')
    else if (!caps.blockMergeNodes.includes(recipe.blockNode)) {
      problems.push(`The server has no ${recipe.blockNode} node.`)
    }
    if (blockInputs) {
      const bad = Object.keys(recipe.blocks).filter((k) => !blockInputs.includes(k))
      if (bad.length > 0) problems.push(`These block names are not inputs of ${recipe.blockNode}: ${bad.slice(0, 5).join(', ')}.`)
    }
    for (const [k, v] of Object.entries(recipe.blocks)) {
      if (!Number.isFinite(v) || v < 0 || v > 1) problems.push(`Block "${k}" must be between 0 and 1.`)
    }
  }
  if (!Number.isFinite(recipe.ratio) || recipe.ratio < 0 || recipe.ratio > 1) problems.push('The ratio must be between 0 and 1.')
  if (!Number.isFinite(recipe.multiplier) || recipe.multiplier < 0 || recipe.multiplier > 2) {
    problems.push('The multiplier must be between 0 and 2.')
  }
  if (!recipe.outputName || recipe.outputName.trim() === '') problems.push('Enter a name for the merged file.')
  return problems
}

/** Copy of the recipe where every block input it leaves out gets `recipe.ratio` (ComfyUI needs every input of the node). */
export function completeBlocks(recipe: MergeRecipe, blockInputs: string[]): MergeRecipe {
  const blocks = { ...recipe.blocks }
  for (const k of blockInputs) if (!(k in blocks)) blocks[k] = recipe.ratio
  return { ...recipe, blocks }
}

export function buildMergeGraph(recipe: MergeRecipe): BuiltGraph {
  const prompt: Record<string, unknown> = {}
  let next = 1
  const add = (class_type: string, inputs: Record<string, unknown>): string => {
    const id = String(next++)
    prompt[id] = { class_type, inputs }
    return id
  }
  const isCkpt = recipe.kind === 'checkpoint'
  const ratio = r4(clamp(recipe.ratio, 0, 1))
  const multiplier = r4(clamp(recipe.multiplier, 0, 2))
  const needC = recipe.mode === 'addDifference' && recipe.c !== null

  const load = (name: string): string =>
    isCkpt
      ? add('CheckpointLoaderSimple', { ckpt_name: name })
      : add('UNETLoader', { unet_name: name, weight_dtype: recipe.weightDtype })
  const a = load(recipe.a)
  const b = load(recipe.b)
  const c = needC ? load(recipe.c as string) : null
  const model = (id: string): Link => [id, 0]
  const clipOf = (id: string): Link => [id, 1]
  const vaeOf = (id: string): Link => [id, 2]

  let merged: Link
  if (recipe.mode === 'addDifference' && c) {
    const diff = add('ModelMergeSubtract', { model1: model(b), model2: model(c), multiplier })
    merged = [add('ModelMergeAdd', { model1: model(a), model2: [diff, 0] }), 0]
  } else if (recipe.mode === 'blocks') {
    const inputs: Record<string, unknown> = { model1: model(a), model2: model(b) }
    for (const [k, v] of Object.entries(recipe.blocks)) inputs[k] = r4(clamp(v, 0, 1))
    merged = [add(recipe.blockNode ?? 'ModelMergeBlocks', inputs), 0]
  } else {
    merged = [add('ModelMergeSimple', { model1: model(a), model2: model(b), ratio }), 0]
  }

  const name = sanitizeOutputName(recipe.outputName)
  if (!isCkpt) {
    const subfolder = 'diffusion_models'
    const filenamePrefix = `${subfolder}/${name}`
    add('ModelSave', { model: merged, filename_prefix: filenamePrefix })
    return { prompt, saveNode: 'ModelSave', outputSubfolder: subfolder, filenamePrefix }
  }

  let clip: Link
  if (recipe.clip === 'a') clip = clipOf(a)
  else if (recipe.clip === 'b') clip = clipOf(b)
  else if (recipe.mode === 'addDifference' && c) {
    const diff = add('CLIPMergeSubtract', { clip1: clipOf(b), clip2: clipOf(c), multiplier })
    clip = [add('CLIPMergeAdd', { clip1: clipOf(a), clip2: [diff, 0] }), 0]
  } else {
    clip = [add('CLIPMergeSimple', { clip1: clipOf(a), clip2: clipOf(b), ratio }), 0]
  }
  const vae = recipe.vae === 'b' ? vaeOf(b) : vaeOf(a)
  const subfolder = 'checkpoints'
  const filenamePrefix = `${subfolder}/${name}`
  add('CheckpointSave', { model: merged, clip, vae, filename_prefix: filenamePrefix })
  return { prompt, saveNode: 'CheckpointSave', outputSubfolder: subfolder, filenamePrefix }
}
