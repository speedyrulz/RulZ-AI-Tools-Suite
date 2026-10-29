import type { DraftStats } from '@shared/types'
import type { GenTimings } from './sse'

/** Turns shorter than this give too noisy a speed to keep as the no-drafting baseline. */
const MIN_BASELINE_TOKENS = 16

/** Milliseconds per token of a turn generated without drafting, or undefined when it drafted or was too short to trust. */
export function noDraftMsPerToken(gen: GenTimings): number | undefined {
  if (gen.drafted || gen.generated < MIN_BASELINE_TOKENS || gen.genMs <= 0) return undefined
  return gen.genMs / gen.generated
}

/**
 * Generation time one turn saved by drafting.
 *
 * With a measured no-drafting speed (ms per token): generated × baseline − genMs.
 * Without one: accepted × (genMs / passes), where passes = generated − accepted, since each pass of the
 * model yields one token of its own plus the drafts it accepted. A pass costs more than a plain token
 * (it also drafts and verifies), so this is an upper bound.
 */
export function turnSavedMs(gen: GenTimings, baselineMs?: number): { savedMs: number; measured: boolean } {
  const accepted = gen.accepted ?? 0
  if (baselineMs) return { savedMs: gen.generated * baselineMs - gen.genMs, measured: true }
  const passes = gen.generated - accepted
  if (!accepted || passes <= 0) return { savedMs: 0, measured: false }
  return { savedMs: accepted * (gen.genMs / passes), measured: false }
}

/** Adds one turn to a task's draft stats; turns that didn't draft leave them unchanged. */
export function addDraft(total: DraftStats | undefined, gen: GenTimings, baselineMs?: number): DraftStats | undefined {
  if (!gen.drafted) return total
  const { savedMs, measured } = turnSavedMs(gen, baselineMs)
  const t = total ?? { drafted: 0, accepted: 0, generated: 0, genMs: 0, savedMs: 0, measured: true }
  return {
    drafted: t.drafted + gen.drafted,
    accepted: t.accepted + (gen.accepted ?? 0),
    generated: t.generated + gen.generated,
    genMs: t.genMs + gen.genMs,
    savedMs: t.savedMs + savedMs,
    measured: t.measured && measured
  }
}
