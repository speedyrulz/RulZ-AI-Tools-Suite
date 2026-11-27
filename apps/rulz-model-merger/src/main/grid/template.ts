// Sample grids: reads a user's ComfyUI workflow (API format) and builds one grid cell's prompt from it. Pure, no I/O.
import type { GridTemplateInfo } from '@shared/grid'
import type { MergeRecipe } from '@shared/merge'
import { buildMergeModelSubgraph } from '../merge/graph'

type Node = { class_type: string; inputs: Record<string, unknown> }
type Link = [string, number]
type Nodes = Record<string, Node>

const LOADER_CLASSES = ['UNETLoader', 'CheckpointLoaderSimple']
const IMAGE_OUTPUT_CLASSES = ['SaveImage', 'PreviewImage']
const TEXT_KEYS = ['text', 'prompt', 'text_g', 't5xxl']
/** Prompt inputs an encoder may carry side by side (SDXL: text_g + text_l, Flux: clip_l + t5xxl). */
const SIBLING_KEYS = ['text', 'prompt', 'text_g', 'text_l', 'clip_l', 't5xxl']

/** The main prompt input plus its siblings on the same node that hold a string or a link. */
function promptInputNames(n: Node | undefined, main: string): string[] {
  if (!n) return [main]
  return [main, ...SIBLING_KEYS.filter((k) => k !== main && (typeof n.inputs[k] === 'string' || isLink(n.inputs[k])))]
}
/** Inputs the prompt walk does not follow (they never lead to the text). */
const SKIP_KEYS = new Set(['model', 'clip', 'vae'])
const SEED_KEYS = ['seed', 'noise_seed']

const UI_FORMAT_ERROR =
  'This is a workflow file, not the API format. In ComfyUI use Workflow > Export (API) and pick that file.'

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function isLink(v: unknown): v is Link {
  return Array.isArray(v) && v.length === 2 && typeof v[0] === 'string' && typeof v[1] === 'number'
}

function fileName(path: string): string {
  const parts = path.split(/[\\/]/)
  return parts[parts.length - 1] || path
}

/** Numeric-aware id order ("5:1" < "10"). */
function compareIds(a: string, b: string): number {
  const pa = a.split(':')
  const pb = b.split(':')
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const x = pa[i]
    const y = pb[i]
    if (x === undefined) return -1
    if (y === undefined) return 1
    const nx = Number(x)
    const ny = Number(y)
    if (Number.isFinite(nx) && Number.isFinite(ny) && nx !== ny) return nx - ny
    if (!(Number.isFinite(nx) && Number.isFinite(ny)) && x !== y) return x < y ? -1 : 1
  }
  return 0
}

function parseNodes(json: unknown): Nodes | null {
  if (!isRecord(json)) return null
  const out: Nodes = {}
  const keys = Object.keys(json)
  if (keys.length === 0) return null
  for (const k of keys) {
    const n = json[k]
    if (!isRecord(n) || typeof n.class_type !== 'string' || !isRecord(n.inputs)) return null
    out[k] = { class_type: n.class_type, inputs: n.inputs }
  }
  return out
}

/** Every node that feeds `starts` (the starts included), following links backward. `cut` is one input whose link is ignored. */
function upstream(nodes: Nodes, starts: string[], cut?: { id: string; inputs: string[] }): Set<string> {
  const seen = new Set<string>()
  const stack = [...starts]
  while (stack.length > 0) {
    const id = stack.pop() as string
    if (seen.has(id) || !nodes[id]) continue
    seen.add(id)
    for (const [k, v] of Object.entries(nodes[id].inputs)) {
      if (!isLink(v)) continue
      if (cut && cut.id === id && cut.inputs.includes(k)) continue
      stack.push(v[0])
    }
  }
  return seen
}

/** Every node fed by the loader's MODEL output (index 0) and, from there, by any output. The loader is not included. */
function downstreamOfModel(nodes: Nodes, loaderId: string): Set<string> {
  const seen = new Set<string>()
  let queue: string[] = []
  const eat = (from: string, index: number | null): string[] => {
    const out: string[] = []
    for (const [nid, n] of Object.entries(nodes)) {
      for (const v of Object.values(n.inputs)) {
        if (isLink(v) && v[0] === from && (index === null || v[1] === index)) {
          out.push(nid)
          break
        }
      }
    }
    return out
  }
  queue = eat(loaderId, 0)
  while (queue.length > 0) {
    const next: string[] = []
    for (const id of queue) {
      if (seen.has(id) || id === loaderId) continue
      seen.add(id)
      next.push(...eat(id, null))
    }
    queue = next
  }
  return seen
}

/**
 * True when the loader's MODEL output reaches a sampler-like node through `model` inputs only (model patches, LoRA
 * loaders, guiders). A loader whose model only feeds e.g. a detector, whose masks or images later reach the sampler
 * (SAM 3 crop -> VAEEncode -> latent), does not count.
 */
function modelReachesSampler(nodes: Nodes, loaderId: string): boolean {
  const chain = new Set<string>([loaderId])
  let grew = true
  while (grew) {
    grew = false
    for (const [nid, n] of Object.entries(nodes)) {
      if (chain.has(nid)) continue
      const fedModel = Object.entries(n.inputs).some(
        ([k, v]) => /^model\d*$/.test(k) && isLink(v) && chain.has(v[0]) && (v[0] !== loaderId || v[1] === 0)
      )
      if (!fedModel) continue
      if (isSamplerLike(n) || /sampler/i.test(n.class_type)) return true
      chain.add(nid)
      grew = true
    }
  }
  return false
}

function isSamplerLike(n: Node): boolean {
  if (isLink(n.inputs.positive) || isLink(n.inputs.guider)) return true
  return /guider/i.test(n.class_type) && (isLink(n.inputs.model) || isLink(n.inputs.conditioning))
}

function samplerEntryLink(n: Node): Link | null {
  if (isLink(n.inputs.positive)) return n.inputs.positive
  if (/guider/i.test(n.class_type) && isLink(n.inputs.conditioning)) return n.inputs.conditioning
  return null
}

/**
 * Breadth-first walk backward from `startId` for the first encoder-like node: one whose text / prompt (then text_g,
 * t5xxl) input is a string or a link. With `stopAtZeroOut` the walk does not pass through ConditioningZeroOut.
 */
function findEncoder(nodes: Nodes, startId: string, stopAtZeroOut: boolean): { id: string; input: string } | null {
  const seen = new Set<string>()
  let queue = [startId]
  while (queue.length > 0) {
    const next: string[] = []
    for (const id of queue) {
      if (seen.has(id) || !nodes[id]) continue
      seen.add(id)
      if (stopAtZeroOut && nodes[id].class_type === 'ConditioningZeroOut') continue
      for (const key of TEXT_KEYS) {
        const v = nodes[id].inputs[key]
        if (typeof v === 'string' || isLink(v)) return { id, input: key }
      }
      for (const [k, v] of Object.entries(nodes[id].inputs)) if (isLink(v) && !SKIP_KEYS.has(k)) next.push(v[0])
    }
    queue = next
  }
  return null
}

/** The literal of a primitive string node (PrimitiveString*, or a node whose only input is a string value), else null. */
function primitiveString(n: Node | undefined): string | null {
  if (!n) return null
  const v = n.inputs.value
  if (typeof v !== 'string') return null
  if (/^PrimitiveString/.test(n.class_type)) return v
  const keys = Object.keys(n.inputs)
  return keys.length === 1 && keys[0] === 'value' ? v : null
}

/** Follows a linked seed to the node that holds the literal number. */
function resolveSeed(nodes: Nodes, id: string, seen: Set<string>): { id: string; input: string } | null {
  if (seen.has(id) || !nodes[id]) return null
  seen.add(id)
  for (const key of [...SEED_KEYS, 'value']) {
    const v = nodes[id].inputs[key]
    if (typeof v === 'number') return { id, input: key }
    if (isLink(v) && key !== 'value') {
      const r = resolveSeed(nodes, v[0], seen)
      if (r) return r
    }
  }
  return null
}

export function inspectTemplate(json: unknown, path: string): GridTemplateInfo {
  const info: GridTemplateInfo = {
    path,
    name: fileName(path),
    error: null,
    warnings: [],
    modelLoader: null,
    promptInput: null,
    seedInputs: [],
    imageOutputs: [],
    defaultPrompt: null
  }
  const fail = (msg: string): GridTemplateInfo => {
    info.error = msg
    return info
  }

  if (isRecord(json) && Array.isArray(json.nodes) && 'links' in json) return fail(UI_FORMAT_ERROR)
  const nodes = parseNodes(json)
  if (!nodes) {
    return fail(
      'This file is not a ComfyUI workflow in API format. In ComfyUI use Workflow > Export (API) and pick that file.'
    )
  }
  const ids = Object.keys(nodes)

  const allOutputs = ids.filter((id) => IMAGE_OUTPUT_CLASSES.includes(nodes[id].class_type))
  if (allOutputs.length === 0) {
    return fail('The workflow has no SaveImage or PreviewImage node, so there is nothing to show in the grid.')
  }

  // Model loader: the one whose MODEL output feeds a sampler. Other loaders (a side branch) are ignored.
  const loaders = ids.filter((id) => LOADER_CLASSES.includes(nodes[id].class_type))
  if (loaders.length === 0) {
    const others = [
      ...new Set(
        ids
          .map((id) => nodes[id].class_type)
          .filter((c) => /loader/i.test(c) && !LOADER_CLASSES.includes(c) && !/lora|vae|clip|controlnet|upscale/i.test(c))
      )
    ]
    let msg = 'The workflow has no UNETLoader or CheckpointLoaderSimple node. Only those two model loaders are supported.'
    if (others.length > 0) msg += ` Found ${others.join(', ')}, which is not supported.`
    return fail(msg)
  }
  let loaderId = loaders[0]
  if (loaders.length > 1) {
    const feeding = loaders.filter((id) => {
      const d = [...downstreamOfModel(nodes, id)]
      return modelReachesSampler(nodes, id) && d.some((x) => allOutputs.includes(x))
    })
    if (feeding.length !== 1) {
      return fail(
        `The workflow has ${loaders.length} model loaders (${loaders.map((id) => `${nodes[id].class_type} ${id}`).join(', ')}) and the app cannot tell which one feeds the sampler. Remove the extra loaders from the template.`
      )
    }
    loaderId = feeding[0]
    for (const id of loaders) {
      if (id !== loaderId) {
        info.warnings.push(`The template also has ${nodes[id].class_type} ${id}, which does not feed the sampler and is ignored.`)
      }
    }
  }
  info.modelLoader = { id: loaderId, classType: nodes[loaderId].class_type }

  // One image output: downstream of the model, SaveImage first, then the lowest id.
  const fed = downstreamOfModel(nodes, loaderId)
  const candidates = allOutputs.filter((id) => fed.has(id))
  if (candidates.length === 0) {
    return fail('No SaveImage or PreviewImage node is fed by the model loader, so there is nothing to show in the grid.')
  }
  candidates.sort((a, b) => {
    const sa = nodes[a].class_type === 'SaveImage' ? 0 : 1
    const sb = nodes[b].class_type === 'SaveImage' ? 0 : 1
    return sa !== sb ? sa - sb : compareIds(a, b)
  })
  const primary = candidates[0]
  info.imageOutputs = [primary]
  if (allOutputs.length > 1) {
    info.warnings.push(`Only ${primary} (${nodes[primary].class_type}) is rendered; other outputs are left out.`)
  }

  // Positive prompt: the encoder reached backward from the first sampler that feeds the primary output.
  const up = upstream(nodes, [primary])
  // Only samplers on the path from the chosen loader to the primary output count.
  const onPath = ids.filter((id) => samplerEntryLink(nodes[id]) !== null && up.has(id) && fed.has(id))
  const hits = onPath.flatMap((id) => {
    const link = samplerEntryLink(nodes[id]) as Link
    const found = findEncoder(nodes, link[0], false)
    return found ? [{ id, link, found }] : []
  })
  if (hits.length === 0) {
    return fail('The app could not find the positive prompt: no sampler leads back to a text prompt node.')
  }
  const keyOf = (h: { found: { id: string; input: string } }): string => `${h.found.id}|${h.found.input}`
  let chosen = hits[0]
  if (new Set(hits.map(keyOf)).size > 1) {
    // Several passes with different prompts: the first pass (no other sampler of the path feeds it) decides.
    const firsts = hits.filter((h) => !hits.some((o) => o.id !== h.id && upstream(nodes, [h.id]).has(o.id)))
    if (firsts.length === 0 || new Set(firsts.map(keyOf)).size !== 1) {
      return fail('The workflow has several samplers with different prompts and the app cannot tell which one the grid should change.')
    }
    chosen = firsts[0]
  }
  const samplerId = chosen.id
  const entry = chosen.link
  info.promptInput = { id: chosen.found.id, input: chosen.found.input, classType: nodes[chosen.found.id].class_type }
  const promptValue = nodes[info.promptInput.id].inputs[info.promptInput.input]
  if (typeof promptValue === 'string') {
    info.defaultPrompt = promptValue
  } else if (isLink(promptValue)) {
    const src = nodes[promptValue[0]]
    const lit = primitiveString(src)
    if (lit !== null) info.defaultPrompt = lit
    else {
      info.warnings.push(`The prompt comes from ${src ? src.class_type : 'another node'}; the grid replaces it with your prompts.`)
    }
  }

  // Seeds: only those that still run once the prompt link is cut and the unused branches are dropped.
  const live = upstream(nodes, [primary], {
    id: info.promptInput.id,
    inputs: promptInputNames(nodes[info.promptInput.id], info.promptInput.input)
  })
  const seen = new Set<string>()
  for (const id of ids) {
    if (!live.has(id)) continue
    for (const key of SEED_KEYS) {
      const v = nodes[id].inputs[key]
      const hit = typeof v === 'number' ? { id, input: key } : isLink(v) ? resolveSeed(nodes, v[0], new Set()) : null
      if (hit && !seen.has(`${hit.id}\u0000${hit.input}`)) {
        seen.add(`${hit.id}\u0000${hit.input}`)
        info.seedInputs.push(hit)
      }
    }
  }

  // Warnings
  if (ids.some((id) => live.has(id) && /lora/i.test(nodes[id].class_type))) {
    info.warnings.push('The template has its own LoRA loaders. They run after the merge and are applied on top of it.')
  }
  const neg = nodes[samplerId].inputs.negative
  if (isLink(neg) && !(neg[0] === entry[0] && neg[1] !== entry[1])) {
    const negFound = findEncoder(nodes, neg[0], true)
    if (negFound && negFound.id === info.promptInput.id && negFound.input === info.promptInput.input) {
      info.warnings.push('The negative prompt uses the same text node as the positive prompt, so changing the prompt changes both.')
    }
  }
  if (info.modelLoader.classType === 'CheckpointLoaderSimple') {
    info.warnings.push('The template loads a checkpoint, so only checkpoint merges fit it.')
  } else {
    info.warnings.push('The template loads a diffusion model (UNETLoader), so only diffusion-model merges fit it.')
  }
  return info
}

export function checkTemplateForRecipe(info: GridTemplateInfo, recipe: MergeRecipe): string | null {
  if (info.error) return info.error
  if (!info.modelLoader) return 'The template has no model loader.'
  const wantClass = recipe.kind === 'checkpoint' ? 'CheckpointLoaderSimple' : 'UNETLoader'
  if (info.modelLoader.classType !== wantClass) {
    return recipe.kind === 'checkpoint'
      ? 'The template loads a diffusion model (UNETLoader), but this merge makes a checkpoint. Use a template with a Load Checkpoint node.'
      : 'The template loads a checkpoint (CheckpointLoaderSimple), but this merge makes a diffusion model. Use a template with a Load Diffusion Model node.'
  }
  return null
}

/**
 * One grid cell's prompt: the template with its model loader replaced by the merge, the prompt and seeds set, the
 * primary output turned into a PreviewImage, and every node that does not feed that output removed.
 * `recipe` has the cell's value applied.
 */
export function buildGridPrompt(
  template: Record<string, unknown>,
  info: GridTemplateInfo,
  recipe: MergeRecipe,
  prompt: string,
  seed: number
): Record<string, unknown> {
  const problem = checkTemplateForRecipe(info, recipe)
  if (problem) throw new Error(problem)
  const loader = info.modelLoader as { id: string; classType: string }
  const out = JSON.parse(JSON.stringify(template)) as Nodes
  if (!out[loader.id]) throw new Error(`The template has no node ${loader.id}; load the template again.`)
  delete out[loader.id]

  let prefix = 'mm'
  while (Object.keys(out).some((k) => k.startsWith(prefix)) || loader.id.startsWith(prefix)) prefix += '_'
  const sub = buildMergeModelSubgraph(recipe, prefix)

  const replacement = (index: number): Link => {
    if (index === 0) return sub.model
    if (index === 1 && sub.clip) return sub.clip
    if (index === 2 && sub.vae) return sub.vae
    throw new Error(`A node in the template reads output ${index} of the model loader, which a ${recipe.kind} merge does not provide.`)
  }
  for (const n of Object.values(out)) {
    for (const [k, v] of Object.entries(n.inputs)) {
      if (isLink(v) && v[0] === loader.id) n.inputs[k] = [...replacement(v[1])]
    }
  }

  if (info.promptInput) {
    const n = out[info.promptInput.id]
    if (!n) throw new Error(`The template has no node ${info.promptInput.id}; load the template again.`)
    for (const key of promptInputNames(n, info.promptInput.input)) n.inputs[key] = prompt
  }
  for (const s of info.seedInputs) if (out[s.id]) out[s.id].inputs[s.input] = seed

  const primary = info.imageOutputs[0]
  if (!primary || !out[primary]) throw new Error('The template has no image output; load the template again.')
  if (out[primary].class_type === 'SaveImage') {
    out[primary] = { class_type: 'PreviewImage', inputs: { images: out[primary].inputs.images } }
  }

  const all: Nodes = { ...out, ...(sub.nodes as Nodes) }
  const keep = upstream(all, [primary])
  const result: Record<string, unknown> = {}
  for (const [id, n] of Object.entries(all)) if (keep.has(id)) result[id] = n
  return result
}
