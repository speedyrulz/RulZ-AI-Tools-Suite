// LoRA + LoRA merges (milestone 5). Computed by the app itself (worker thread, CPU), not by ComfyUI.
//
// Math: a LoRA layer changes a weight by  delta = up @ down * scale,  scale = alpha / rank (1 when there is no alpha).
// - concat (exact): the merged layer stacks the inputs along the rank:
//     up' = [w1*s1*up1 | w2*s2*up2 | ...]  (out x R),  down' = [down1; down2; ...]  (R x in),  alpha' = R
//   so  up' @ down' = sum_i w_i * s_i * up_i @ down_i  exactly (up to the saved dtype's rounding).
// - shrink: the stacked layer is reduced to rank min(k, R) with an SVD of the low-rank factors (never the full
//   out x in matrix). The kept share of the layer's energy (sum of kept squared singular values / sum of all) is
//   reported per layer.
// Conv layers (down [r, in, kh, kw], up [out, r, 1, 1]) are handled as 2-D by flattening down to [r, in*kh*kw].
// Supported: plain LoRA keys (lora_up/lora_down, lora_A/lora_B, lora.up/lora.down) with optional .alpha. Not
// supported (refused with a reason): LoKr, LoHa, DoRA (dora_scale), lora_mid (LoCon tucker), full diffs (.diff), and
// up weights with a kernel larger than 1x1.

/** Most input LoRAs one merge may combine. */
export const MAX_LORA_MERGE_INPUTS = 10

export type LoraMergeMode = 'concat' | 'shrink'
/** 'source' = the dtype of the first input's up weights. */
export type LoraMergeDtype = 'source' | 'bf16' | 'fp16' | 'fp32'

export interface LoraMergeInput {
  /** Name as the server lists it (relative to models/loras). */
  name: string
  /** Multiplies this LoRA's change; may be negative. -10..10. */
  weight: number
}

export interface LoraMergeRecipe {
  inputs: LoraMergeInput[]
  mode: LoraMergeMode
  /** Target rank for 'shrink', 1..1024. Ignored for 'concat'. */
  rank: number
  dtype: LoraMergeDtype
  /** File name stem; letters, digits, - _ . and spaces only (same rule as model merges). */
  outputName: string
}

export const DEFAULT_LORA_MERGE_RECIPE: LoraMergeRecipe = {
  inputs: [],
  mode: 'concat',
  rank: 64,
  dtype: 'source',
  outputName: 'merged-lora'
}

/** What the app read from one input LoRA's header. */
export interface LoraMergeInputInfo {
  name: string
  readable: boolean
  error: string | null
  sizeBytes: number | null
  /** "LoRA", "LoKr", "LoHa", "DoRA", "mixed" or "unknown". */
  format: string
  keyStyle: 'comfy' | 'kohya' | 'diffusers' | 'unknown'
  /** Layers (modules) with a usable up/down pair. */
  modules: number
  /** Of those, text-encoder layers. */
  textEncoderModules: number
  /** Smallest and largest rank over its layers; null when none. */
  rank: { min: number; max: number } | null
  /** dtype of its up weights ("BF16", "F16", "F32"); "mixed" or null. */
  dtype: string | null
  baseModelHint: string | null
  /** false when the file cannot be merged (format not supported, no layers, dtype not readable). */
  supported: boolean
  unsupportedReason: string | null
}

export interface LoraMergeCheck {
  inputs: LoraMergeInputInfo[]
  /** Block the merge. */
  problems: string[]
  warnings: string[]
  /** Layers in the result. */
  totalModules: number
  /** Layers that two or more inputs change. */
  sharedModules: number
  /** Rank of the result's layers (after shrink when the recipe asks for it), null when unknown. */
  outputRank: { min: number; max: number } | null
  /** Estimated size of the saved file in bytes, null when unknown. */
  estimatedBytes: number | null
}

export interface LoraMergeInspectRequest {
  names: string[]
  mode: LoraMergeMode
  rank: number
  dtype: LoraMergeDtype
}

export type LoraMergeStatus = 'running' | 'saving' | 'done' | 'error' | 'cancelled'

export interface LoraMergeOutput {
  /** Final WSL path in <models>/loras/merged/. */
  wslPath: string
  /** Name as ComfyUI lists it, e.g. "merged/my-lora.safetensors". */
  loraName: string
  sizeBytes: number
  modules: number
  rank: { min: number; max: number }
  /** shrink only: share of energy kept, mean and lowest over the layers (0..1). */
  energyKept: { mean: number; min: number } | null
  verify: { ok: boolean; message: string }
  sidecarPath: string | null
}

export interface LoraMergeJob {
  id: string
  recipe: LoraMergeRecipe
  status: LoraMergeStatus
  createdAt: number
  finishedAt: number | null
  /** Layers done / total while running. */
  progress: { done: number; total: number } | null
  error: string | null
  output: LoraMergeOutput | null
}

/** Stored in the saved file's metadata (key "rulz_model_merger", JSON) and in <file>.recipe.json. */
export interface LoraMergeRecord {
  app: 'RulZ Model Merger'
  appVersion: string
  kind: 'lora-merge'
  createdAt: string
  recipe: LoraMergeRecipe
  /** SHA-256 of each input's header, in input order (null if unreadable). */
  inputHeaderSha256: Array<string | null>
}
