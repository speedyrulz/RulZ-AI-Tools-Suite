import { ArrowLeft, CircleCheck, CirclePause, CircleDashed, LoaderCircle, Play, Square } from 'lucide-react'
import { useCallback, useMemo, useState } from 'react'
import { nextOutput, type BenchVote, type WritingBenchRun } from '@shared/writingBench'
import { pairToShow, pairView, runProgress } from '../../lib/wbenchRunner'
import { useWbench } from '../../stores/wbench'
import { Badge, Button, ProgressBar } from '../ui'
import { Scoreboard } from './Scoreboard'
import { VotePanel } from './VotePanel'

export function StatusBadge({ status }: { status: WritingBenchRun['status'] }) {
  if (status === 'running') {
    return (
      <Badge tone="accent">
        <LoaderCircle size={12} className="animate-spin" aria-hidden /> Running
      </Badge>
    )
  }
  if (status === 'done') {
    return (
      <Badge tone="ok">
        <CircleCheck size={12} aria-hidden /> Done
      </Badge>
    )
  }
  if (status === 'stopped') {
    return (
      <Badge tone="warn">
        <CirclePause size={12} aria-hidden /> Stopped
      </Badge>
    )
  }
  return (
    <Badge>
      <CircleDashed size={12} aria-hidden /> Not started
    </Badge>
  )
}

/** One run: progress and Stop while it generates, the blind vote panel, and the scoreboard. */
export function RunView({ run }: { run: WritingBenchRun }) {
  const activeId = useWbench((s) => s.activeId)
  const stopping = useWbench((s) => s.stopping)
  const live = useWbench((s) => s.live)
  const running = useWbench((s) => s.running)
  const shownPairId = useWbench((s) => s.shownPairId)
  const open = useWbench((s) => s.open)
  const stop = useWbench((s) => s.stop)
  const resume = useWbench((s) => s.resume)
  const vote = useWbench((s) => s.vote)
  const showPair = useWbench((s) => s.showPair)
  const [showScores, setShowScores] = useState(false)

  const active = activeId === run.id
  const p = runProgress(run)
  const pair = pairToShow(run, shownPairId)
  const view = useMemo(() => (pair ? pairView(run, pair) : null), [run, pair])
  const canResume = !active && !!nextOutput(run)
  const allVoted = p.pairs > 0 && p.voted === p.pairs
  // Scores open by themselves only when the run is over and every pair is voted: during a run more pairs follow.
  const scoresOpen = allVoted && !active
  const hasNext = run.pairs.some((x) => x.vote === null && x.id !== pair?.id)
  const current = active ? run.outputs.find((o) => o.key === live?.key) : undefined
  const currentPrompt = current ? run.prompts.find((x) => x.id === current.promptId) : undefined

  const onVote = useCallback(
    (v: BenchVote) => {
      if (!pair) return
      // Keep this pair on screen after the vote so its names can be read.
      showPair(pair.id)
      vote(pair.id, v)
    },
    [pair, showPair, vote]
  )
  const onNext = useCallback(() => showPair(null), [showPair])

  return (
    <div className="mx-auto max-w-5xl space-y-5 px-6 py-6">
      <div className="flex flex-wrap items-center gap-3">
        <Button size="sm" variant="ghost" onClick={() => open(null)}>
          <ArrowLeft size={14} aria-hidden /> All runs
        </Button>
        <h1 className="min-w-0 flex-1 truncate font-[family-name:var(--font-display)] text-lg font-semibold">{run.label || 'Untitled run'}</h1>
        <StatusBadge status={run.status} />
        {canResume && (
          <Button variant="primary" size="sm" disabled={running} onClick={() => void resume(run.id)}>
            <Play size={14} aria-hidden /> Resume
          </Button>
        )}
      </div>

      {active ? (
        <div className="rounded-lg border border-accent/40 bg-accent-soft/40 p-3">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <div className="flex min-w-0 items-center gap-2 text-[13px] font-medium">
              <LoaderCircle size={14} className="shrink-0 animate-spin text-accent" aria-hidden />
              <span className="truncate tabular-nums">
                {current ? `Writing output ${p.finished + 1} of ${p.total}` : p.finished === 0 ? 'Loading the first model…' : 'Getting the next output ready…'}
              </span>
            </div>
            <div className="flex shrink-0 gap-1.5">
              <Button size="sm" disabled={!!stopping} onClick={() => stop('after')}>
                Stop after this
              </Button>
              <Button size="sm" variant="danger" disabled={stopping === 'now'} onClick={() => stop('now')}>
                <Square size={12} aria-hidden /> Stop now
              </Button>
            </div>
          </div>
          <ProgressBar value={p.total ? p.finished / p.total : 0} />
          <div className="mt-1.5 flex flex-wrap gap-x-3 text-[11.5px] text-muted tabular-nums">
            {currentPrompt && <span>{currentPrompt.title}</span>}
            {current && live && <span>{live.text.length.toLocaleString('en-US')} characters so far</span>}
            {stopping && <span className="text-warn">{stopping === 'now' ? 'Stopping…' : 'Stopping after this output…'}</span>}
          </div>
          {live && live.text && <p className="mt-2 line-clamp-2 text-[12px] text-faint italic">…{live.text.slice(-200)}</p>}
        </div>
      ) : (
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-[12.5px] text-muted tabular-nums">
          <span>
            {p.finished} of {p.total} outputs written{p.failed > 0 ? ` (${p.failed} failed)` : ''}
          </span>
        </div>
      )}
      <div className="text-[12.5px] text-muted tabular-nums">
        {p.voted} of {p.pairs} pairs voted{active ? ' so far' : ''}
      </div>

      {view ? (
        <VotePanel key={view.pair.id} view={view} onVote={onVote} onNext={onNext} hasNext={hasNext} />
      ) : (
        <div className="rounded-lg border border-dashed border-border px-4 py-6 text-center text-[13px] text-faint">
          {p.pairs === 0
            ? 'Pairs to vote on appear as outputs finish. Each pair is two models answering the same prompt with the same seed.'
            : allVoted
              ? active
                ? 'You have voted on every pair so far. More appear as outputs finish.'
                : 'You have voted on every pair.'
              : 'No pair is waiting.'}
        </div>
      )}

      <section>
        <div className="mb-2 flex items-center gap-3">
          <h2 className="text-[11px] font-semibold tracking-wide text-faint uppercase">Scoreboard</h2>
          {!scoresOpen && p.voted > 0 && (
            <Button size="sm" variant="ghost" onClick={() => setShowScores((v) => !v)}>
              {showScores ? 'Hide scores' : 'Show scores'}
            </Button>
          )}
        </div>
        {scoresOpen || (showScores && p.voted > 0) ? (
          <Scoreboard run={run} />
        ) : (
          <p className="text-[12.5px] text-faint">
            {p.voted === 0 ? 'Scores appear after your first vote.' : 'Hidden while you vote, so the lead does not sway you. Show it any time, or finish the pairs.'}
          </p>
        )}
      </section>
    </div>
  )
}
