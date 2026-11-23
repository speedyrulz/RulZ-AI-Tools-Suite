import { CircleAlert, Cloud, CloudOff, TimerReset } from 'lucide-react'
import { chipFor } from '../../lib/remote'
import { cn } from '../../lib/format'
import { useApp } from '../../stores/app'
import { isBusy, useRemote } from '../../stores/remote'
import { Button, Tooltip } from '../ui'
import { useNow } from './useNow'

/**
 * Top-bar chip, on every page: what the running rentals cost per hour and so far, or what stopped ones cost for
 * their disks. Turns amber with Keep running when the app is about to stop a rental.
 */
export function CostChip() {
  const state = useRemote((s) => s.state)
  const run = useRemote((s) => s.run)
  const keeping = useRemote((s) => isBusy(s, 'remote:keepRunning', s.state ? (chipFor(s.state.rentals, 0).countdown?.rentalId ?? 0) : 0))
  const setPage = useApp((s) => s.setPage)
  const now = useNow(1000)
  if (!state) return null
  const chip = chipFor(state.rentals, now)
  const unsettled = state.intents.length > 0
  if (chip.kind === 'none' && !unsettled) return null

  const lines = [...chip.lines]
  if (unsettled) lines.push('A rent request is not settled: an instance may have been rented. Open the vast.ai page.')

  if (chip.countdown) {
    const cd = chip.countdown
    return (
      <div className="flex shrink-0 items-center gap-1.5 rounded-lg border border-warn/40 bg-warn/10 py-0.5 pr-0.5 pl-2.5 text-xs text-warn">
        <button className="flex items-center gap-1.5 hover:underline focus-visible:underline" onClick={() => setPage('vast')}>
          <TimerReset size={14} />
          <span className="tabular-nums">Stopping in {cd.seconds} s{cd.more > 0 ? ` (and ${cd.more} more)` : ''}</span>
          <span className="text-warn/80 tabular-nums">· {chip.text}</span>
        </button>
        <Button size="sm" disabled={keeping} onClick={() => void run('remote:keepRunning', cd.rentalId)}>
          Keep running
        </Button>
      </div>
    )
  }

  const tone = unsettled && chip.kind === 'none' ? 'warn' : chip.kind === 'running' ? 'run' : 'quiet'
  return (
    <Tooltip
      content={
        <div className="space-y-0.5">
          {lines.map((l, i) => (
            <div key={i}>{l}</div>
          ))}
        </div>
      }
    >
      <button
        onClick={() => setPage('vast')}
        aria-label={`vast.ai: ${chip.text || 'rent request not settled'}`}
        className={cn(
          'flex h-9 shrink-0 items-center gap-1.5 rounded-lg border px-2.5 text-xs tabular-nums transition-colors',
          tone === 'run' && 'border-accent/50 bg-accent-soft/60 text-text hover:border-accent',
          tone === 'quiet' && 'border-border bg-panel-2 text-muted hover:text-text',
          tone === 'warn' && 'border-warn/40 bg-warn/10 text-warn'
        )}
      >
        {tone === 'warn' ? <CircleAlert size={14} /> : tone === 'run' ? <Cloud size={14} className="text-accent" /> : <CloudOff size={14} />}
        <span>{chip.text || 'Rent request not settled'}</span>
      </button>
    </Tooltip>
  )
}
