import type { DraftStats } from '@shared/types'
import { formatDuration, formatNumber } from './format'

export interface DraftSummary {
  /** Share of drafted tokens the model accepted, in percent. */
  rate: number
  /** "1.20 s saved", "≤ 1.20 s saved" (an upper bound) or "300 ms slower". */
  saved: string
  /** The counts behind it. */
  detail: string
  /** How the time saved was worked out. */
  how: string
}

/** Speculative decoding (MTP / draft model) for one task, as the computer's Drafts readout shows it; null when the task didn't draft. */
export function draftSummary(draft: DraftStats | undefined): DraftSummary | null {
  if (!draft || !draft.drafted) return null
  const rate = Math.round((draft.accepted / draft.drafted) * 100)
  return {
    rate,
    saved: draft.savedMs >= 0 ? `${draft.measured ? '' : '≤ '}${formatDuration(draft.savedMs)} saved` : `${formatDuration(-draft.savedMs)} slower`,
    detail:
      `${formatNumber(draft.drafted)} tokens drafted, ${formatNumber(draft.accepted)} accepted (${rate}%), ` +
      `over ${formatNumber(draft.generated)} tokens written in ${formatDuration(draft.genMs)}.`,
    how: draft.measured
      ? 'Time saved = tokens written × time per token without drafting (measured on this model) − actual writing time.'
      : 'Time saved ≤ accepted × time per model pass. A pass also drafts and checks the guesses, so this is an upper bound. ' +
        'Turn drafting off for one task in RulZ LLM Suite to measure the plain speed.'
  }
}
