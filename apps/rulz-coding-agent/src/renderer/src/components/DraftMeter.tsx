import { formatDuration } from '../lib/format'
import { useWorkspace } from '../stores/workspace'
import { Tooltip } from './ui'

/** Speculative decoding (MTP / draft model) for the shown task: how many guesses the model kept, and the time that saved. */
export function DraftMeter() {
  const draft = useWorkspace((s) => (s.tasks.find((t) => t.id === s.selectedTaskId) ?? s.tasks[s.tasks.length - 1])?.draft)
  if (!draft || !draft.drafted) return null
  const rate = Math.round((draft.accepted / draft.drafted) * 100)
  const saved = draft.savedMs >= 0 ? `${draft.measured ? '' : '≤ '}${formatDuration(draft.savedMs)} saved` : `${formatDuration(-draft.savedMs)} slower`

  return (
    <Tooltip
      content={
        <div className="space-y-1">
          <p>
            Speculative decoding (MTP) in this task: {draft.drafted.toLocaleString('en-US')} tokens drafted, {draft.accepted.toLocaleString('en-US')} accepted
            ({rate}%), over {draft.generated.toLocaleString('en-US')} tokens written in {formatDuration(draft.genMs)}.
          </p>
          {draft.measured ? (
            <p>Time saved = tokens written × time per token without drafting (measured on this model) − actual writing time.</p>
          ) : (
            <p>
              Time saved ≤ accepted × time per model pass, where a pass is the writing time ÷ (tokens written − accepted). A pass also drafts and checks the
              guesses, so it costs more than a plain token: this is an upper bound. Turn drafting off for one task in RulZ LLM Suite to measure the plain speed.
            </p>
          )}
        </div>
      }
    >
      <div className="flex items-center gap-1.5 text-[11px] text-faint tabular-nums" aria-label={`Drafts ${rate}% accepted, ${saved}`}>
        <span>Drafts</span>
        <span>
          {rate}% · {saved}
        </span>
      </div>
    </Tooltip>
  )
}
