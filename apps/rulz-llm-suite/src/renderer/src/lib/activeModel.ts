import type { LoadedInstance } from '@shared/types'

/**
 * The model to keep active given the instances now listed. `loading` is a model a load from this
 * window is starting: it stays active while its old instance stops for a reload, instead of the
 * active model jumping to another loaded model.
 */
export function pickActive(instances: LoadedInstance[], active: string | null, loading: string | null): string | null {
  const stillThere = active !== null && (active === loading || instances.some((i) => i.modelId === active && i.status !== 'stopping'))
  if (stillThere) return active
  const newest = [...instances].filter((i) => i.status !== 'stopping').sort((a, b) => b.startedAt - a.startedAt)[0]
  return newest?.modelId ?? null
}
