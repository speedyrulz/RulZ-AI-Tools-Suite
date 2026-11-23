import { ChevronRight, ListChecks, Play, Square, Trash2 } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  cleanBatchOptions,
  DEFAULT_BATCH_OPTIONS,
  isFinished,
  planBatch,
  summarize,
  type BatchScore
} from '@shared/batch'
import type { BatchItem, BatchItemStatus, BatchOptions, BatchRun, BatchState, RePracticeTaskStats } from '@shared/types'
import { toast } from 'sonner'
import { errorMessage, invoke, on } from '../lib/api'
import { cn, formatDuration, relativeTime } from '../lib/format'
import { useWorkspace } from '../stores/workspace'
import { Badge, Button, Input, Spinner, Switch } from './ui'

const BATCH_OPTS_KEY = 'rulz.practiceBatchOptions'
export const OLD_BATCH_MSG = 'Update RulZ Coding Agent on the computer to 0.22.0 or later to use batch runs from the phone.'

export function isOldBatchError(e: unknown): boolean {
  return /batch:|re:setKit/.test(errorMessage(e))
}

/** A computer running an older RulZ Coding Agent doesn't know the RE or batch requests. */
export function friendlyReError(e: unknown): string {
  const m = errorMessage(e)
  if (/re:(overview|start)/.test(m)) return 'Update RulZ Coding Agent on the computer to 0.18.0 or later to use reverse engineering from the phone.'
  if (isOldBatchError(e)) return OLD_BATCH_MSG
  return m
}

const STATUS: Record<BatchItemStatus, { label: string; tone: 'neutral' | 'accent' | 'warn' | 'danger' | 'ok' }> = {
  pending: { label: 'Not run', tone: 'neutral' },
  running: { label: 'Running', tone: 'accent' },
  solved: { label: 'Solved', tone: 'ok' },
  wrong: { label: 'Wrong', tone: 'danger' },
  'no-answer': { label: 'No answer', tone: 'danger' },
  timeout: { label: 'Time limit', tone: 'warn' },
  error: { label: 'Error', tone: 'neutral' },
  stopped: { label: 'Stopped', tone: 'neutral' }
}

const pct = (s: BatchScore): string => (s.rate === null ? '—' : `${Math.round(s.rate * 100)}%`)

const baseName = (p: string): string => p.split(/[\\/]/).filter(Boolean).at(-1) ?? p

function loadSavedOptions(): BatchOptions {
  try {
    const raw = localStorage.getItem(BATCH_OPTS_KEY)
    if (raw) return cleanBatchOptions(JSON.parse(raw) as Partial<BatchOptions>)
  } catch {
    // Ignore localStorage failures.
  }
  return DEFAULT_BATCH_OPTIONS
}

function saveOptions(opts: BatchOptions): void {
  try {
    localStorage.setItem(BATCH_OPTS_KEY, JSON.stringify(opts))
  } catch {
    // Ignore localStorage failures.
  }
}

/** Loads `batch:state` when the sheet opens and listens to live `batch:changed` events from the computer. */
export function useBatchState(
  open: boolean,
  onProgress?: () => void
): { state: BatchState | null; supported: boolean; setState: (s: BatchState) => void } {
  const [state, setState] = useState<BatchState | null>(null)
  const [supported, setSupported] = useState(true)
  const lastFinished = useRef<number | null>(null)
  const onProgressRef = useRef(onProgress)
  onProgressRef.current = onProgress

  useEffect(() => {
    if (!open) return
    invoke('batch:state')
      .then((s) => {
        setState(s)
        setSupported(true)
        const active = s.runs.find((r) => r.id === s.activeId)
        lastFinished.current = active ? summarize(active).finished : null
      })
      .catch((e) => {
        if (isOldBatchError(e)) {
          setSupported(false)
          setState(null)
        } else {
          toast.error(friendlyReError(e))
        }
      })
  }, [open])

  useEffect(() => {
    return on('batch:changed', (next) => {
      setState(next)
      setSupported(true)
      const active = next.runs.find((r) => r.id === next.activeId)
      const finished = active ? summarize(active).finished : null
      if (open && lastFinished.current !== null && finished !== lastFinished.current) {
        onProgressRef.current?.()
      }
      lastFinished.current = finished
    })
  }, [open])

  return { state, supported, setState }
}

/** Batch run form and live progress while a run is active. */
export function BatchRunBox({
  tasks,
  family,
  record,
  busy,
  state,
  onStateChange
}: {
  tasks: RePracticeTaskStats[]
  family: string
  record: boolean
  busy: boolean
  state: BatchState | null
  onStateChange: (s: BatchState) => void
}) {
  const project = useWorkspace((s) => s.project)
  const [opts, setOpts] = useState<BatchOptions>(loadSavedOptions)
  const [label, setLabel] = useState('')
  const [starting, setStarting] = useState(false)
  const [stoppingMode, setStoppingMode] = useState<'after' | 'now' | null>(null)

  const options = useMemo(() => cleanBatchOptions({ ...opts, family }), [opts, family])
  const solvedIds = useMemo(() => new Set(tasks.filter((t) => t.solved > 0).map((t) => t.id)), [tasks])
  const planned = useMemo(() => planBatch(tasks, options, solvedIds), [tasks, options, solvedIds])
  const taskCount = options.attempts > 0 ? planned.length / options.attempts : 0

  const set = <K extends keyof BatchOptions>(key: K, value: BatchOptions[K]): void => {
    setOpts((prev) => {
      const next = cleanBatchOptions({ ...prev, [key]: value })
      saveOptions(next)
      return next
    })
  }

  const active = state?.runs.find((r) => r.id === state.activeId)
  if (active) {
    const s = summarize(active)
    const current = active.items.find((i) => i.status === 'running')
    const stopping = state?.stopping ?? stoppingMode
    const stop = async (mode: 'after' | 'now'): Promise<void> => {
      setStoppingMode(mode)
      try {
        onStateChange(await invoke('batch:stop', mode))
      } catch (e) {
        toast.error(friendlyReError(e))
      } finally {
        setStoppingMode(null)
      }
    }
    return (
      <div className="space-y-2.5 rounded-xl border border-accent/40 bg-accent-soft/40 p-3.5">
        <div className="flex items-center gap-2 text-[14px] font-medium tabular-nums">
          <Spinner className="shrink-0 text-accent" />
          <span className="min-w-0 flex-1 truncate">
            Batch run{active.label ? ` “${active.label}”` : ''}: task {Math.min(s.finished + 1, s.total)} of {s.total}
          </span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-track">
          <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${s.total ? Math.round((s.finished / s.total) * 100) : 0}%` }} />
        </div>
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-[13px] text-muted tabular-nums">
          <span>{current ? `${current.taskId}${active.options.attempts > 1 ? ` (try ${current.attempt})` : ''}` : 'Starting the next task…'}</span>
          <span>
            {s.solved} of {s.scored} solved{s.rate !== null ? ` (${pct(s)})` : ''}
          </span>
          {stopping && <span className="text-warn">{stopping === 'now' ? 'Stopping…' : 'Stopping after this task…'}</span>}
        </div>
        <div className="flex gap-2 pt-0.5">
          <Button size="sm" className="flex-1" disabled={!!stopping} onClick={() => void stop('after')}>
            Stop after this task
          </Button>
          <Button size="sm" variant="danger" className="flex-1" disabled={stopping === 'now'} onClick={() => void stop('now')}>
            <Square size={13} /> Stop now
          </Button>
        </div>
      </div>
    )
  }

  const startBatch = async (): Promise<void> => {
    if (!project || !planned.length) return
    setStarting(true)
    try {
      const next = await invoke('batch:start', project.path, label.trim(), options, record)
      onStateChange(next)
      setLabel('')
    } catch (e) {
      toast.error(friendlyReError(e))
    } finally {
      setStarting(false)
    }
  }

  return (
    <div className="space-y-3 rounded-xl border border-border bg-panel-2/40 p-3.5">
      <div>
        <div className="text-[15px] font-medium">Batch run</div>
        <p className="mt-0.5 text-[13px] text-muted">
          Runs the tasks one after another without you, grades each, and keeps a score per run. Tools that ask for approval are approved automatically in batch runs.
        </p>
      </div>
      <div className="grid grid-cols-3 gap-2">
        <label className="block space-y-1">
          <span className="block text-[12px] text-muted">At most tasks</span>
          <Input
            type="number"
            inputMode="numeric"
            min={0}
            max={100000}
            placeholder="all"
            className="tabular-nums"
            value={opts.maxTasks ? String(opts.maxTasks) : ''}
            onChange={(e) => set('maxTasks', Math.max(0, parseInt(e.target.value, 10) || 0))}
          />
        </label>
        <label className="block space-y-1">
          <span className="block text-[12px] text-muted">Tries per task</span>
          <Input
            type="number"
            inputMode="numeric"
            min={1}
            max={10}
            className="tabular-nums"
            value={String(opts.attempts)}
            onChange={(e) => set('attempts', Math.min(10, Math.max(1, parseInt(e.target.value, 10) || 1)))}
          />
        </label>
        <label className="block space-y-1">
          <span className="block text-[12px] text-muted">Minutes / task</span>
          <Input
            type="number"
            inputMode="numeric"
            min={0}
            max={1440}
            placeholder="none"
            className="tabular-nums"
            value={opts.minutesPerTask ? String(opts.minutesPerTask) : ''}
            onChange={(e) => set('minutesPerTask', Math.min(1440, Math.max(0, parseInt(e.target.value, 10) || 0)))}
          />
        </label>
      </div>
      <label className="flex items-center justify-between gap-3 text-[14px]">
        <span>Only unsolved</span>
        <Switch checked={opts.onlyUnsolved} onCheckedChange={(v) => set('onlyUnsolved', v)} />
      </label>
      <Input
        value={label}
        maxLength={80}
        placeholder="Label, e.g. base or lora-v2 (optional)"
        onChange={(e) => setLabel(e.target.value)}
      />
      <div className="flex items-center justify-between gap-3">
        <span className="text-[13px] text-faint tabular-nums">
          {planned.length === 0
            ? '0 tasks'
            : `${taskCount} task${taskCount === 1 ? '' : 's'}${options.attempts > 1 ? ` × ${options.attempts} tries (${planned.length})` : ''}${family ? ` · only ${family}` : ''}`}
        </span>
        <Button variant="primary" disabled={!project || !planned.length || busy || starting} onClick={() => void startBatch()}>
          {starting ? <Spinner /> : <Play size={15} />} Start batch run
        </Button>
      </div>
    </div>
  )
}

/** Expandable compact scoreboard of saved batch runs. */
export function BatchScoreboard({ state, onStateChange }: { state: BatchState | null; onStateChange: (s: BatchState) => void }) {
  if (!state || state.runs.length === 0) return null
  return (
    <div className="space-y-2.5">
      <div className="flex items-center gap-1.5 text-[13px] font-medium text-muted">
        <ListChecks size={15} /> Scoreboard
      </div>
      <div className="space-y-2">
        {state.runs.map((run) => (
          <RunCard key={run.id} run={run} active={state.activeId === run.id} onStateChange={onStateChange} />
        ))}
      </div>
    </div>
  )
}

function RunCard({ run, active, onStateChange }: { run: BatchRun; active: boolean; onStateChange: (s: BatchState) => void }) {
  const [open, setOpen] = useState(false)
  const [confirmRemove, setConfirmRemove] = useState(false)
  const [removing, setRemoving] = useState(false)
  const s = useMemo(() => summarize(run), [run])
  const multi = run.options.attempts > 1

  const removeRun = async (): Promise<void> => {
    setRemoving(true)
    try {
      onStateChange(await invoke('batch:delete', run.id))
    } catch (e) {
      toast.error(friendlyReError(e))
      setRemoving(false)
      setConfirmRemove(false)
    }
  }

  return (
    <div className="overflow-hidden rounded-xl border border-border bg-panel">
      <button
        type="button"
        className="w-full px-3.5 py-3 text-left active:bg-hover"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <div className="flex items-center gap-2">
          <ChevronRight size={15} className={cn('shrink-0 text-faint transition-transform', open && 'rotate-90')} />
          <span className="min-w-0 flex-1 truncate text-[14px] font-medium">{run.label || 'Unlabelled'}</span>
          {active ? (
            <Badge tone="accent">running</Badge>
          ) : run.status === 'stopped' ? (
            <Badge tone="warn">stopped</Badge>
          ) : (
            <Badge tone="neutral">done</Badge>
          )}
          <span className="shrink-0 text-[14px] tabular-nums">
            <span className="font-semibold">{pct(s)}</span>{' '}
            <span className="text-muted">
              {s.solved}/{s.scored}
            </span>
          </span>
        </div>
        <div className="mt-1 flex flex-wrap gap-x-2 gap-y-0.5 pl-5 text-[12px] text-faint tabular-nums">
          <span className="truncate">{baseName(run.modelName || 'Unknown model')}</span>
          <span>· {relativeTime(run.createdAt)}</span>
          <span>
            · {s.tasksSolved}/{s.tasksScored} tasks solved
          </span>
          <span>
            · {s.medianRounds === null ? '—' : `${Math.round(s.medianRounds)} rounds`} · {formatDuration(s.medianMs)}
          </span>
        </div>
      </button>
      {open && (
        <div className="space-y-3 border-t border-border bg-panel-2/30 px-3.5 py-3">
          {s.families.length > 0 && (
            <div className="flex flex-wrap gap-1.5 text-[12px]">
              {s.families.map((f) => (
                <span key={f.family} className="rounded-lg border border-border bg-panel px-2.5 py-1 tabular-nums">
                  {f.family} <span className="font-medium">{pct(f.score)}</span>{' '}
                  <span className="text-faint">
                    {f.score.solved}/{f.score.scored}
                  </span>
                </span>
              ))}
            </div>
          )}
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-[12px] text-faint tabular-nums">
            {(['solved', 'wrong', 'no-answer', 'timeout', 'error', 'stopped', 'pending'] as const)
              .filter((k) => s.counts[k] > 0)
              .map((k) => (
                <span key={k}>
                  {STATUS[k].label}: {s.counts[k]}
                </span>
              ))}
            <span>{run.options.minutesPerTask ? `${run.options.minutesPerTask} min/task` : 'No time limit'}</span>
            {multi && <span>{run.options.attempts} tries/task</span>}
            {run.options.family && <span>Only {run.options.family}</span>}
            {run.options.onlyUnsolved && <span>Only unsolved</span>}
          </div>
          <div className="max-h-72 divide-y divide-border overflow-y-auto rounded-lg border border-border bg-panel">
            {run.items.map((item: BatchItem, idx) => {
              const st = STATUS[item.status]
              return (
                <div key={`${item.taskId}-${item.attempt}-${idx}`} className="px-3 py-2">
                  <div className="flex items-center gap-2">
                    <span className="min-w-0 flex-1 truncate font-mono text-[13px]">
                      {item.taskId}
                      {multi && <span className="font-sans text-[12px] text-faint tabular-nums"> · try {item.attempt}</span>}
                    </span>
                    <Badge tone={st.tone}>{st.label}</Badge>
                  </div>
                  <div className="mt-0.5 flex flex-wrap gap-x-2.5 gap-y-0.5 text-[12px] text-faint tabular-nums">
                    <span>
                      {item.family} · {item.tier}
                    </span>
                    {item.toolRounds !== undefined && <span>· {item.toolRounds} rounds</span>}
                    {isFinished(item.status) && <span>· {formatDuration(item.ms)}</span>}
                  </div>
                  {(item.answer !== undefined || item.expected !== undefined) && item.status !== 'solved' && (
                    <div className="mt-0.5 font-mono text-[12px] break-all text-faint">
                      answered {item.answer ?? '—'}
                      {item.expected !== undefined ? ` · expected ${item.expected}` : ''}
                    </div>
                  )}
                  {item.error && <div className="mt-0.5 text-[12px] text-danger">{item.error}</div>}
                </div>
              )
            })}
          </div>
          {!confirmRemove ? (
            <div className="flex justify-end">
              <Button size="sm" disabled={active || removing} onClick={() => setConfirmRemove(true)}>
                <Trash2 size={14} /> Remove
              </Button>
            </div>
          ) : (
            <div className="flex items-center justify-end gap-2">
              <span className="mr-auto text-[13px] text-muted">Remove this saved run?</span>
              <Button size="sm" disabled={removing} onClick={() => setConfirmRemove(false)}>
                Cancel
              </Button>
              <Button size="sm" variant="danger" disabled={removing} onClick={() => void removeRun()}>
                {removing ? <Spinner /> : <Trash2 size={14} />} Remove
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
