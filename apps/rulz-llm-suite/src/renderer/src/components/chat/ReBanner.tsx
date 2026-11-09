import { Binary, CircleDot } from 'lucide-react'
import type { Conversation, ReOutcome } from '@shared/types'
import { cn } from '../../lib/format'
import { invoke } from '../../lib/api'
import { useChat } from '../../stores/chat'
import { Badge, Tooltip } from '../ui'

export const OUTCOMES: { value: Exclude<ReOutcome, 'unrated'>; label: string; tone: string }[] = [
  { value: 'solved', label: 'Solved', tone: 'border-ok/40 bg-ok/15 text-ok' },
  { value: 'partial', label: 'Partial', tone: 'border-warn/40 bg-warn/15 text-warn' },
  { value: 'failed', label: 'Failed', tone: 'border-danger/40 bg-danger/15 text-danger' }
]

/** Grade buttons; clicking the current grade clears it. */
export function OutcomeButtons({
  id,
  outcome,
  size = 'md',
  onChange
}: {
  id: string
  outcome: ReOutcome
  size?: 'sm' | 'md'
  /** Grades something other than a chat (an API session); chats are graded through the chat store. */
  onChange?: (outcome: ReOutcome) => void
}) {
  const setReOutcome = useChat((s) => s.setReOutcome)
  const set = (o: ReOutcome): void => (onChange ? onChange(o) : void setReOutcome(id, o))
  return (
    <div className="flex items-center gap-1">
      {OUTCOMES.map((o) => (
        <button
          key={o.value}
          onClick={() => set(outcome === o.value ? 'unrated' : o.value)}
          className={cn(
            'rounded border font-medium transition-colors',
            size === 'sm' ? 'px-1.5 py-px text-[10.5px]' : 'px-2 py-0.5 text-[11px]',
            outcome === o.value ? o.tone : 'border-border text-muted hover:bg-hover hover:text-text'
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

/** Strip under an RE session's header: the target, whether data is recorded, and its grade. */
export function ReBanner({ conv }: { conv: Conversation }) {
  const setReCollect = useChat((s) => s.setReCollect)
  const re = conv.re
  if (!re) return null
  const p = re.practice
  return (
    <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1 border-b border-border bg-panel/60 px-5 py-1.5 text-[11.5px]">
      <span className="flex min-w-0 items-center gap-1.5 text-muted">
        <Binary size={13} className="shrink-0 text-accent" />
        <span className="truncate font-mono" title={re.target}>
          {re.target}
        </span>
      </span>
      {p && (
        <Badge tone="accent">
          Practice {p.family} · {p.tier}
        </Badge>
      )}
      {re.clone && (
        <Tooltip content={`Writing the ${re.clone.scope === 'full' ? 'whole-program' : 'piece'} clone${re.clone.language ? ` (${re.clone.language})` : ''} here. Click to open.`}>
          <button className="flex min-w-0 items-center gap-1 text-muted hover:text-text" onClick={() => void invoke('shell:openPath', re.clone!.outputDir)}>
            <Badge tone="accent">Clone</Badge>
            <span className="truncate font-mono">→ {re.clone.outputDir}</span>
          </button>
        </Tooltip>
      )}
      {p?.answer !== undefined && (
        <span className="text-faint">
          answer <span className="font-mono text-text">{p.answer || '—'}</span>
          {p.expected !== undefined && re.outcome !== 'solved' && (
            <>
              {' '}
              · expected <span className="font-mono text-text">{p.expected}</span>
            </>
          )}
        </span>
      )}
      <div className="flex-1" />
      <Tooltip
        content={
          re.collect
            ? 'Recording what the model sees and writes, for fine-tuning. Click to stop.'
            : 'Not recording this session. Click to record it for fine-tuning.'
        }
      >
        <button
          onClick={() => void setReCollect(conv.id, !re.collect)}
          className={cn('flex items-center gap-1 rounded px-1.5 py-0.5 hover:bg-hover', re.collect ? 'text-danger' : 'text-faint')}
        >
          <CircleDot size={12} className={re.collect ? 'animate-pulse' : ''} />
          {re.collect ? 'Recording' : 'Not recording'}
        </button>
      </Tooltip>
      <span className="text-faint">{re.autoGraded ? 'Auto-graded:' : 'Grade:'}</span>
      <OutcomeButtons id={conv.id} outcome={re.outcome} />
    </div>
  )
}
