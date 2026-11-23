import { Binary, CircleDot, ListChecks, SkipForward } from 'lucide-react'
import type { Conversation, ReOutcome } from '@shared/types'
import { cn } from '../../lib/format'
import { invoke } from '../../lib/api'
import { POOL_CHAT_TEXT } from '../../lib/remote'
import { useBatch } from '../../stores/batch'
import { isPoolChat, useChat } from '../../stores/chat'
import { Badge, Button, Tooltip } from '../ui'

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

/**
 * Shown while a batch run writes this chat (in the main process, maybe on a rented GPU): the chat is read-only here,
 * and the task can be skipped. The run goes on with the next task.
 */
export function PoolBanner({ id }: { id: string }) {
  const owned = useChat((s) => isPoolChat(s, id))
  const item = useBatch((s) => s.runs.find((r) => r.id === s.activeId)?.items.find((i) => i.conversationId === id && i.status === 'running'))
  const skip = useBatch((s) => s.skip)
  if (!owned) return null
  return (
    <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1 border-b border-accent/40 bg-accent-soft/50 px-5 py-2 text-[12px]" role="status">
      <ListChecks size={14} className="shrink-0 text-accent" />
      <span className="font-medium">{POOL_CHAT_TEXT}</span>
      <span className="min-w-0 flex-1 truncate text-muted">
        You can read it as it grows. It cannot be edited or added to until the task ends.
        {item?.host ? ` Running on ${item.host}.` : ''}
      </span>
      <Tooltip content="Stops this session. Its task counts as stopped, and the run takes the next task.">
        <Button size="sm" onClick={() => skip(id)}>
          <SkipForward size={13} /> Skip this task
        </Button>
      </Tooltip>
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
