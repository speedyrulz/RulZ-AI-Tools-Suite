import { cn, formatTokens } from '../lib/format'
import { useWorkspace } from '../stores/workspace'
import { Tooltip } from './ui'

/** Same threshold as the main process's small-context warning. */
const SMALL_CONTEXT = 16384

/** How full the model's working memory is, from the last request. */
export function ContextMeter() {
  const context = useWorkspace((s) => s.context)
  if (!context || !context.size) return null
  const share = Math.min(1, context.used / context.size)
  const pct = Math.round(share * 100)
  const small = context.size <= SMALL_CONTEXT

  return (
    <Tooltip
      content={
        <div className="space-y-1">
          <p>
            The AI's working memory: {context.used.toLocaleString('en-US')} of {context.size.toLocaleString('en-US')} tokens
            {context.exact ? ' (counted by RulZ)' : ' (estimated)'}.
          </p>
          <p>When it fills up, I tidy older notes into a short summary so I can keep going.</p>
          {small && <p className="text-warn">This is small. For bigger projects, load the model in RulZ LLM Suite with a context length of 32K or more.</p>}
        </div>
      }
    >
      <div className="flex items-center gap-1.5 text-[11px] text-faint" aria-label={`Memory ${pct}% full`} role="meter" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <span>Memory</span>
        <span className="h-1.5 w-14 overflow-hidden rounded-full bg-track">
          <span
            className={cn('block h-full rounded-full', share >= 0.85 ? 'bg-danger' : share >= 0.6 || small ? 'bg-warn' : 'bg-accent')}
            style={{ width: `${Math.max(3, pct)}%` }}
          />
        </span>
        <span className="tabular-nums">
          {pct}% of {formatTokens(context.size)}
        </span>
      </div>
    </Tooltip>
  )
}
