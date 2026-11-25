import { formatDollars } from '@shared/modes'
import { useWorkspace } from '../stores/workspace'
import { Tooltip } from './ui'

/** What the shown task has cost at OpenRouter so far; nothing for a model that reports no cost. */
export function CostMeter() {
  const cost = useWorkspace((s) => (s.tasks.find((t) => t.id === s.selectedTaskId) ?? s.tasks[s.tasks.length - 1])?.cost)
  if (!cost) return null
  return (
    <Tooltip content={<p>What this task has cost at OpenRouter so far, from OpenRouter's own numbers. Requests that were cut off are not in it; LLM Suite's OpenRouter page has them. Its daily spending limit applies.</p>}>
      <div className="flex items-center gap-1.5 text-[11px] text-faint tabular-nums" aria-label={`Cost ${formatDollars(cost)} at OpenRouter`}>
        <span>Cost</span>
        <span>{formatDollars(cost)}</span>
      </div>
    </Tooltip>
  )
}
