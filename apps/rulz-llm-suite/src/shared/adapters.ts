// LoRA adapters: which ones go with a model, and the strengths a request uses. Pure; shared by main and renderer.
import type { AdapterEntry, InstanceAdapter, LoadAdapter, ModelEntry } from './types'

export const MIN_ADAPTER_SCALE = -4
export const MAX_ADAPTER_SCALE = 4

export function clampScale(v: unknown, fallback = 1): number {
  const n = typeof v === 'number' && Number.isFinite(v) ? v : fallback
  return Math.min(MAX_ADAPTER_SCALE, Math.max(MIN_ADAPTER_SCALE, Math.round(n * 1000) / 1000))
}

/** Adapters offered for a model: the same architecture, anywhere in the models folder. */
export function adapterChoices(model: ModelEntry, adapters: AdapterEntry[]): AdapterEntry[] {
  const arch = model.info?.architecture
  if (!arch || arch === 'unknown') return []
  return adapters.filter((a) => a.architecture === arch)
}

const norm = (s: string): string => s.toLowerCase().replace(/[^a-z0-9]+/g, '')

/**
 * Whether an adapter's recorded base model is this model. 'unknown' when the adapter doesn't say.
 * Compared by name with separators dropped, either containing the other ("Qwen3.5-35B-A3B" fits
 * "Qwen3.5 35B A3B Instruct"), against the model's general.name and its file name.
 */
export function adapterBaseMatch(adapter: AdapterEntry, model: ModelEntry): 'match' | 'mismatch' | 'unknown' {
  const base = norm(adapter.info?.baseModelName ?? '')
  if (!base) return 'unknown'
  const names = [model.info?.name ?? '', model.name].map(norm).filter(Boolean)
  return names.some((n) => n.includes(base) || base.includes(n)) ? 'match' : 'mismatch'
}

/** llama.cpp splits --lora values on commas, so a path with one can't be passed. */
export function adapterPathUsable(path: string): boolean {
  return !path.includes(',')
}

/**
 * The `lora` field for a llama-server request: every adapter the instance loaded, by llama id, at the
 * chat's strength where it sets one, else the default. Null when the instance loaded none.
 */
export function requestLora(loaded: InstanceAdapter[], chat: LoadAdapter[] | null | undefined): { id: number; scale: number }[] | null {
  if (!loaded.length) return null
  return loaded.map((a, id) => {
    const own = chat?.find((c) => c.id === a.id)
    return { id, scale: own ? clampScale(own.scale) : a.scale }
  })
}
