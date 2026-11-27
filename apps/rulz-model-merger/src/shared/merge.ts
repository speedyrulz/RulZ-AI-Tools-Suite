// Contract for checkpoint / diffusion-model merges (milestone 3). Shared by main and renderer.
//
// ComfyUI conventions (checked in comfy_extras/nodes_model_merging.py, ComfyUI 0.46):
// - ModelMergeSimple(model1, model2, ratio) = model1 * ratio + model2 * (1 - ratio). `ratio` is the share of model1 (A).
// - ModelMergeSubtract(model1, model2, multiplier) = (model1 - model2) * multiplier.
// - ModelMergeAdd(model1, model2) = model1 + model2.
// - Block nodes (ModelMergeBlocks, ModelMergeSDXL, ModelMergeFlux1, ...): every FLOAT input is a key prefix of the
//   diffusion model (after "diffusion_model."); the longest matching prefix sets that weight's share of model1.
// - Save nodes write in the loaded weights' format; a quantized model is re-quantized to model1's format on save.
// - extra_data.extra_pnginfo entries are written into the saved safetensors header as JSON strings.
// - LoraLoaderModelOnly(model, lora_name, strength_model) / LoraLoader(model, clip, lora_name, strength_model,
//   strength_clip) add the LoRA as a patch; the save nodes apply the patches before writing. A LoRA weight ComfyUI
//   cannot map to the model is skipped with a log line "lora key not loaded: <key>" (or "NOT LOADED <key>").

/** Where the sources come from: CheckpointLoaderSimple (models/checkpoints) or UNETLoader (models/diffusion_models). */
export type MergeSourceKind = 'checkpoint' | 'diffusion'

/** weighted: A*r + B*(1-r). addDifference: A + (B - C)*m. blocks: per-block share of A through a block merge node.
 *  bake: no merge, only A with the recipe's LoRAs applied. In the other modes the LoRAs are applied to the merge result. */
export type MergeMode = 'weighted' | 'addDifference' | 'blocks' | 'bake'

/** One LoRA baked into the result, applied in list order. */
export interface LoraEntry {
  /** Name exactly as the server lists it (lora_name choices). */
  name: string
  /** strength_model. The UI slider covers -2..2; values from -10 to 10 are accepted. */
  strength: number
  /** strength_clip, checkpoints only (diffusion-model files have no text encoder). */
  clipStrength: number
}

/** Most LoRAs one recipe may bake. */
export const MAX_LORAS = 10

/** Checkpoint merges only: what happens to the text encoder. */
export type ClipChoice = 'merge' | 'a' | 'b'
/** Checkpoint merges only: which VAE is saved. A VAE is never merged. */
export type VaeChoice = 'a' | 'b'
/** UNETLoader weight_dtype. "default" keeps the file's own format. */
export type UnetDtype = 'default' | 'fp8_e4m3fn' | 'fp8_e4m3fn_fast' | 'fp8_e5m2'

export interface MergeRecipe {
  kind: MergeSourceKind
  mode: MergeMode
  /** Names exactly as the server lists them (ckpt_name / unet_name choices). */
  a: string
  b: string
  /** addDifference only: the model subtracted from B. */
  c: string | null
  /** weighted (and the CLIP in other modes): share of A, 0..1. */
  ratio: number
  /** addDifference: multiplier m, 0..2 in the UI. */
  multiplier: number
  /** blocks: the block merge node to use, e.g. "ModelMergeFlux1" or "ModelMergeBlocks". */
  blockNode: string | null
  /** blocks: share of A per block input name of that node. Inputs left out use `ratio`. */
  blocks: Record<string, number>
  clip: ClipChoice
  vae: VaeChoice
  weightDtype: UnetDtype
  /** File name stem; the save node adds _00001_.safetensors. Letters, digits, - _ . and spaces only. */
  outputName: string
  /** LoRAs applied to the result in order. bake needs at least one; at most MAX_LORAS. */
  loras: LoraEntry[]
}

export const DEFAULT_RECIPE: MergeRecipe = {
  kind: 'diffusion',
  mode: 'weighted',
  a: '',
  b: '',
  c: null,
  ratio: 0.5,
  multiplier: 1,
  blockNode: null,
  blocks: {},
  clip: 'merge',
  vae: 'a',
  weightDtype: 'default',
  outputName: 'merged',
  loras: []
}

/** What the app could read from one source file's safetensors header (own server only; files of a URL server are not reachable). */
export interface ModelFileInfo {
  name: string
  readable: boolean
  /** Why it could not be read (URL server, file not found, not a .safetensors file, bad header). */
  error: string | null
  sizeBytes: number | null
  tensorCount: number
  /** Number of tensors per safetensors dtype, e.g. { BF16: 900, I8: 400, F32: 300 }. */
  dtypes: Record<string, number>
  /** True when the file holds quantized weights (integer or fp4/fp8 weights with scale tensors, or a quantization entry in the metadata). */
  quantized: boolean
  /** Short label such as "int8", "fp8", "nvfp4", "mixed", or null. */
  quantLabel: string | null
  /** Prefix in front of the diffusion model's keys, e.g. "model.diffusion_model." or "". */
  keyPrefix: string
}

/** What the app could read from one LoRA's safetensors header (own server only). */
export interface LoraFileInfo {
  name: string
  readable: boolean
  error: string | null
  sizeBytes: number | null
  tensorCount: number
  /** "LoRA", "LoKr", "LoHa", "DoRA", "mixed" or "unknown", from the key suffixes (lora_up/down, lora_A/B, lokr_w1, hada_w1, dora_scale). */
  format: string
  /** Rank of the first low-rank pair, null for LoKr/unknown. */
  rank: number | null
  /** comfy: "diffusion_model.<layer>"; kohya: "lora_unet_<layer_with_underscores>"; diffusers: "transformer.<diffusers name>"
   *  (ComfyUI maps these per model, so the app cannot match them itself); unknown otherwise. */
  keyStyle: 'comfy' | 'kohya' | 'diffusers' | 'unknown'
  /** Number of modules the LoRA changes in the diffusion model. */
  modules: number
  /** Share of those modules that match a layer of model A by name, 0..1. null when A or the LoRA was not read, or the
   *  key style is diffusers/unknown (then only ComfyUI's log after the bake tells). */
  match: number | null
  /** Modules for a text encoder (lora_te*, text_encoder*, te1_/te2_...). Skipped in a diffusion-model bake. */
  textEncoderModules: number
  /** Base model named in the metadata (ss_base_model, modelspec.architecture...), or null. */
  baseModelHint: string | null
}

export interface BlockNodeOption {
  node: string
  /** The node's FLOAT inputs in server order (block prefixes). */
  inputs: string[]
  /** Share of model A's diffusion-model layers that one of the inputs controls, 0..1; null when A's header was not read.
   *  (A node may have more inputs than the model has blocks, e.g. ModelMergeQwenImage has 60 blocks, Qwen-Image 2.1 has 32.) */
  match: number | null
  /** Inputs that match at least one layer of A (all inputs when A's header was not read). Only these need sliders; the
   *  others are still sent, with the recipe's ratio. */
  activeInputs: string[]
}

export interface MergeCheck {
  a: ModelFileInfo | null
  b: ModelFileInfo | null
  c: ModelFileInfo | null
  /** false: the run would fail or produce garbage (blocked). null: not checked (files not readable). */
  compatible: boolean | null
  /** Blocking problems, plain sentences. */
  problems: string[]
  /** Things to know, plain sentences (quantized sources, RAM, formats differ...). */
  warnings: string[]
  /** Block merge nodes on the server, best match first. */
  blockNodes: BlockNodeOption[]
  suggestedBlockNode: string | null
  /** One entry per requested LoRA, in request order. */
  loras: LoraFileInfo[]
  /** Rough RAM the merge needs in WSL, and what is free there (own server only). */
  ramNeededBytes: number | null
  ramFreeBytes: number | null
}

export type MergeJobStatus = 'queued' | 'running' | 'verifying' | 'done' | 'error' | 'cancelled'

export interface MergeVerify {
  ok: boolean
  tensorCount: number
  /** Keys of A (diffusion model part) missing from the output, and output keys not in A. */
  missingKeys: number
  extraKeys: number
  recipeInHeader: boolean
  /** LoRA weights ComfyUI skipped ("lora key not loaded"), text-encoder keys of a diffusion-model bake not counted.
   *  null when not known (server by URL, or no LoRAs). */
  loraKeysNotLoaded: number | null
  /** Up to 5 of those keys. */
  loraKeysNotLoadedSample: string[]
  /** Text-encoder keys skipped because the bake target has no text encoder (diffusion-model file). */
  loraTextEncoderKeysSkipped: number
  message: string
}

export interface MergeOutput {
  /** Relative to the server's output folder, e.g. "diffusion_models/merged_00001_.safetensors". For a URL server only the
   *  prefix is known: "diffusion_models/merged_*.safetensors". */
  relPath: string
  /** Absolute path inside WSL (own server only). */
  wslPath: string | null
  sizeBytes: number | null
  /** Absolute WSL path of "<file>.recipe.json" (own server only). */
  sidecarPath: string | null
  verify: MergeVerify | null
  /** Absolute WSL path after "Move into models". */
  movedTo: string | null
}

export interface MergeJob {
  id: string
  recipe: MergeRecipe
  status: MergeJobStatus
  serverUrl: string
  own: boolean
  promptId: string | null
  createdAt: number
  finishedAt: number | null
  /** Node currently executing (class_type) and its progress, from the websocket. */
  progress: { node: string | null; value: number; max: number } | null
  output: MergeOutput | null
  error: string | null
}

/** Inspect input: which files to look at. */
export interface MergeInspectRequest {
  kind: MergeSourceKind
  mode: MergeMode
  a: string
  b: string
  c: string | null
  /** LoRA names to read, in recipe order. */
  loras: string[]
}

/** The recipe stored in the safetensors header (key "modelmerger_recipe") and in the sidecar. */
export interface RecipeRecord {
  app: 'RulZ Model Merger'
  /** MergeJob.id: proves a saved file belongs to that job before it is verified, given a sidecar or moved. */
  jobId: string
  appVersion: string
  createdAt: string
  serverVersion: string | null
  recipe: MergeRecipe
  /** SHA-256 of each readable source's header bytes (identifies the file without hashing gigabytes). */
  sourceHeaderSha256: { a: string | null; b: string | null; c: string | null; loras: Array<string | null> }
}
