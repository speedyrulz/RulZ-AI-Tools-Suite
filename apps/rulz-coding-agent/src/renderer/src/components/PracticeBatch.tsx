import { ChevronDown, ChevronRight, FolderOpen, LoaderCircle, Play, Square, Trash2 } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'
import type { BatchItem, BatchItemStatus, BatchOptions, BatchRun, BatchState, ReOverview } from '@shared/types'
import { DEFAULT_BATCH_OPTIONS, cleanBatchOptions, familyCounts, nextPracticeTask, planBatch, summarize } from '@shared/batch'
import { CLOUD_BATCH_NOTE, cloudText, RENTAL_NOTE, practiceScope } from '@shared/modes'
import { errorMessage, invoke, on } from '../lib/api'
import { cn, formatDuration, relativeTime } from '../lib/format'
import { activeModelId, useApp } from '../stores/app'
import { useWorkspace } from '../stores/workspace'
import { modelLabel, rentalText } from './common'
import { Badge, Button, Field, IconButton, Input, NumberInput, ProgressBar, Select, Switch } from './ui'

const OPTIONS_KEY = 'rulz-coding-agent.batchOptions'

const STATUS_TONE: Record<BatchItemStatus, 'neutral' | 'ok' | 'warn' | 'danger' | 'accent'> = {
  pending: 'neutral',
  running: 'accent',
  solved: 'ok',
  wrong: 'danger',
  'no-answer': 'warn',
  timeout: 'warn',
  error: 'danger',
  stopped: 'neutral'
}

const STATUS_LABEL: Record<BatchItemStatus, string> = {
  pending: 'pending',
  running: 'running',
  solved: 'solved',
  wrong: 'wrong',
  'no-answer': 'no answer',
  timeout: 'timed out',
  error: 'error',
  stopped: 'stopped'
}

const pct = (rate: number | null): string => (rate === null ? '—' : `${Math.round(rate * 100)}%`)

function loadOptions(): BatchOptions {
  try {
    const raw = localStorage.getItem(OPTIONS_KEY)
    if (raw) return cleanBatchOptions(JSON.parse(raw) as Partial<BatchOptions>)
  } catch {
    // no saved options
  }
  return DEFAULT_BATCH_OPTIONS
}

/** The batch runs and the one going now; follows the main process. */
export function useBatchState(open: boolean): { state: BatchState | null; error: string | null; set: (s: BatchState) => void } {
  const [state, setState] = useState<BatchState | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    if (!open) return
    let live = true
    invoke('batch:state')
      .then((s) => {
        if (live) {
          setState(s)
          setError(null)
        }
      })
      .catch((e) => live && setError(errorMessage(e)))
    const off = on('batch:changed', (s) => setState(s))
    return () => {
      live = false
      off()
    }
  }, [open])
  return { state, error, set: setState }
}

/** Practice tab, top part: kit folder, family tiles, start next task, batch run, scoreboard. */
export function PracticePanel({
  data,
  busy,
  record,
  open,
  family,
  onFamily,
  onReload,
  onStartTask
}: {
  data: ReOverview
  busy: boolean
  record: boolean
  open: boolean
  family: string
  onFamily: (family: string) => void
  onReload: () => void
  onStartTask: (taskId: string) => void
}) {
  const { state, error: stateError, set } = useBatchState(open)
  const active = state?.activeId ? (state.runs.find((r) => r.id === state.activeId) ?? null) : null
  const tasks = data.practice.tasks
  const tiles = useMemo(() => familyCounts(tasks), [tasks])
  const next = useMemo(() => nextPracticeTask(tasks), [tasks])
  return (
    <div className="space-y-4">
      <KitFolder practice={data.practice} locked={!!active} onReload={onReload} />
      {tasks.length > 0 && (
        <>
          {/* Which model the counts are for: one model with a RulZ LLM Suite that counts per model, else all of them together. */}
          <div className="text-[12px] font-medium text-muted" data-testid="practice-scope">
            {practiceScope(data.practice, modelLabel)}
          </div>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {tiles.map((t) => (
              <button
                key={t.family}
                onClick={() => onFamily(family === t.family ? '' : t.family)}
                className={cn(
                  'rounded-lg border px-3 py-2 text-left transition-colors',
                  family === t.family ? 'border-accent/50 bg-accent-soft' : 'border-border bg-panel hover:bg-hover'
                )}
              >
                <div className={cn('truncate text-[13px] font-medium', family === t.family && 'text-accent')}>{t.family}</div>
                <div className="text-[11.5px] tabular-nums text-muted">
                  {t.solved}/{t.total} solved · {t.tried} tried{t.failed !== undefined && <> · {t.failed} failed</>}
                </div>
              </button>
            ))}
          </div>
          <div className="flex items-center gap-3 text-[13px]">
            <span className="tabular-nums text-muted">{tasks.length} tasks</span>
            <div className="flex-1" />
            {next && (
              <span className="min-w-0 truncate font-mono text-[11.5px] text-faint" title="The next task: not solved yet, fewest tries first">
                next: {next.id}
              </span>
            )}
            <Button disabled={!next || busy || !!active} onClick={() => next && onStartTask(next.id)}>
              <Play size={14} /> Start next task
            </Button>
          </div>
          {state ? (
            <BatchBox data={data} record={record} busy={busy} active={active} state={state} family={family} onState={set} />
          ) : stateError ? (
            <div className="rounded-lg border border-warn/40 bg-warn/10 p-3 text-[13px]">{stateError}</div>
          ) : (
            <div className="flex items-center gap-2 text-[13px] text-muted">
              <LoaderCircle size={14} className="animate-spin" /> Loading batch runs…
            </div>
          )}
          {state && <Scoreboard state={state} onState={set} />}
        </>
      )}
    </div>
  )
}

function KitFolder({ practice, locked, onReload }: { practice: ReOverview['practice']; locked: boolean; onReload: () => void }) {
  const [value, setValue] = useState(practice.kitDir)
  const [saving, setSaving] = useState(false)
  useEffect(() => setValue(practice.kitDir), [practice.kitDir])
  const save = async (dir: string): Promise<void> => {
    const next = dir.trim()
    if (!next || next === practice.kitDir) return
    setSaving(true)
    try {
      await invoke('re:setKit', next)
      onReload()
    } catch (e) {
      toast.error(errorMessage(e))
      setValue(practice.kitDir)
    } finally {
      setSaving(false)
    }
  }
  const pick = async (): Promise<void> => {
    try {
      const p = await invoke('dialog:pickFolder', { title: 'Choose the practice kit folder', defaultPath: practice.kitDir || undefined })
      if (p) {
        setValue(p)
        await save(p)
      }
    } catch (e) {
      toast.error(errorMessage(e))
    }
  }
  return (
    <Field label="Kit folder" hint="Shared with RulZ LLM Suite: changing it changes it there too.">
      <div className="flex items-center gap-2">
        <Input
          className="font-mono text-xs"
          value={value}
          disabled={locked || saving}
          placeholder="C:\path\to\practice kit"
          onChange={(e) => setValue(e.target.value)}
          onBlur={() => void save(value)}
          onKeyDown={(e) => e.key === 'Enter' && void save(value)}
        />
        <IconButton label="Choose folder" disabled={locked || saving} onClick={() => void pick()}>
          {saving ? <LoaderCircle size={16} className="animate-spin" /> : <FolderOpen size={16} />}
        </IconButton>
      </div>
      {practice.error && <div className="rounded-lg border border-warn/40 bg-warn/10 p-3 text-[13px]">{practice.error}</div>}
    </Field>
  )
}

function BatchBox({
  data,
  record,
  busy,
  active,
  state,
  family,
  onState
}: {
  data: ReOverview
  record: boolean
  busy: boolean
  active: BatchRun | null
  state: BatchState
  family: string
  onState: (s: BatchState) => void
}) {
  const project = useWorkspace((s) => s.project)
  const modelId = useApp((s) => activeModelId(s.settings, s.connection))
  const rental = useApp((s) => s.connection.models.find((m) => m.id === modelId)?.rental)
  const cloud = useApp((s) => s.connection.models.find((m) => m.id === modelId)?.cloud)
  const [options, setOptions] = useState<BatchOptions>(() => ({ ...loadOptions(), family }))
  const [label, setLabel] = useState('')
  const [starting, setStarting] = useState(false)
  /** The yes to send a run to a rented computer: asked for every run, never remembered. */
  const [rentalYes, setRentalYes] = useState(false)
  useEffect(() => setRentalYes(false), [modelId])
  // A tile click picks the family for the run too.
  useEffect(() => setOptions((o) => ({ ...o, family })), [family])
  const patch = (p: Partial<BatchOptions>): void => {
    const next = { ...options, ...p }
    setOptions(next)
    try {
      localStorage.setItem(OPTIONS_KEY, JSON.stringify(next))
    } catch {
      // not saved
    }
  }
  const families = useMemo(() => [...new Set(data.practice.tasks.map((t) => t.family))].sort(), [data])
  const planned = useMemo(() => {
    const solved = new Set(data.practice.tasks.filter((t) => t.solved > 0).map((t) => t.id))
    return planBatch(data.practice.tasks, options, solved).length
  }, [data, options])

  const startRun = async (): Promise<void> => {
    if (!project) return
    setStarting(true)
    try {
      onState(await invoke('batch:start', project.path, label.trim(), cleanBatchOptions(options), record, !!(rental || cloud) && rentalYes))
    } catch (e) {
      toast.error(errorMessage(e))
    } finally {
      setStarting(false)
    }
  }
  const stop = async (mode: 'after' | 'now'): Promise<void> => {
    try {
      onState(await invoke('batch:stop', mode))
    } catch (e) {
      toast.error(errorMessage(e))
    }
  }

  if (active) return <Progress run={active} stopping={state.stopping} onStop={(m) => void stop(m)} />
  return (
    <div className="space-y-3 rounded-lg border border-border bg-panel p-3">
      <div className="text-[13px] font-medium">Batch run</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
        <Field label="At most" hint="tasks">
          <NumberInput
            className="w-full"
            label="At most this many tasks"
            value={options.maxTasks || null}
            placeholder="all"
            min={1}
            onChange={(v) => patch({ maxTasks: v ?? 0 })}
          />
        </Field>
        <Field label="Tries per task">
          <NumberInput className="w-full" label="Tries per task" value={options.attempts} min={1} max={10} onChange={(v) => patch({ attempts: v ?? 1 })} />
        </Field>
        <Field label="Minutes per task" hint="0 = no limit">
          <NumberInput className="w-full" label="Minutes per task" value={options.minutesPerTask} min={0} max={1440} onChange={(v) => patch({ minutesPerTask: v ?? 0 })} />
        </Field>
        <Field label="Family">
          <Select value={options.family} onChange={(e) => onFamilyChange(e.target.value)}>
            <option value="">Every family</option>
            {families.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Label" hint="optional">
          <Input value={label} maxLength={60} placeholder="base, lora-v2" onChange={(e) => setLabel(e.target.value)} />
        </Field>
        <Field label="Only unsolved">
          <Switch checked={options.onlyUnsolved} onCheckedChange={(v) => patch({ onlyUnsolved: v })} />
        </Field>
      </div>
      {(rental || cloud) && (
        <div className="space-y-1.5 rounded-lg border border-warn/40 bg-warn/10 p-3 text-[13px]">
          <div>
            {cloud ? (
              <>
                <Badge tone="warn">{cloudText({ cloud })}</Badge> {modelLabel(modelId)} runs on OpenRouter. {CLOUD_BATCH_NOTE}
              </>
            ) : (
              <>
                <Badge tone="warn">{rentalText({ rental })}</Badge> {modelLabel(modelId)} runs on a rented computer. {RENTAL_NOTE.replace('This model runs on a rented computer. ', '')}
              </>
            )}
          </div>
          <label className="flex items-center gap-2">
            <Switch checked={rentalYes} onCheckedChange={setRentalYes} label={cloud ? 'Yes, run this batch on OpenRouter' : 'Yes, run this batch on the rented computer'} /> {cloud ? 'Yes, run this batch on OpenRouter' : 'Yes, run this batch on the rented computer'}
          </label>
        </div>
      )}
      <div className="flex items-center gap-3">
        <span className="text-[11.5px] text-faint">Tools that ask for approval are approved automatically in batch runs.</span>
        <div className="flex-1" />
        <span className="text-[13px] tabular-nums text-muted">{planned} {planned === 1 ? 'task' : 'tasks'}</span>
        <Button variant="primary" disabled={!planned || busy || starting || !project || (!!(rental || cloud) && !rentalYes)} onClick={() => void startRun()}>
          {starting ? <LoaderCircle size={14} className="animate-spin" /> : <Play size={14} />} Start batch run
        </Button>
      </div>
    </div>
  )
  function onFamilyChange(f: string): void {
    patch({ family: f })
  }
}

function Progress({ run, stopping, onStop }: { run: BatchRun; stopping: BatchState['stopping']; onStop: (mode: 'after' | 'now') => void }) {
  const s = summarize(run)
  const current = run.items.find((i) => i.status === 'running')
  const index = current ? run.items.indexOf(current) + 1 : s.finished
  // Re-render each second so the elapsed time moves.
  const [, tick] = useState(0)
  useEffect(() => {
    const t = setInterval(() => tick((n) => n + 1), 1000)
    return () => clearInterval(t)
  }, [])
  return (
    <div className="space-y-3 rounded-lg border border-accent/40 bg-panel p-3">
      <div className="flex items-center gap-2 text-[13px]">
        <LoaderCircle size={14} className="animate-spin text-accent" />
        <span className="font-medium">{run.label || 'Batch run'} is running</span>
        <span className="ml-auto tabular-nums text-muted">
          task {Math.min(index, s.total)} of {s.total}
        </span>
      </div>
      <ProgressBar value={s.total ? s.finished / s.total : 0} />
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-muted">
        {current ? (
          <span>
            <span className="font-mono text-text">{current.taskId}</span> · try {current.attempt}
            {current.startedAt ? <span className="tabular-nums"> · {formatDuration(Date.now() - current.startedAt)}</span> : null}
          </span>
        ) : (
          <span>Starting the next task…</span>
        )}
        <span className="tabular-nums">
          solved {s.solved} of {s.scored} scored ({pct(s.rate)})
        </span>
      </div>
      <div className="flex items-center gap-2">
        {stopping && <span className="text-[12px] text-warn">{stopping === 'after' ? 'Stopping after this task…' : 'Stopping…'}</span>}
        <div className="flex-1" />
        <Button disabled={!!stopping} onClick={() => onStop('after')}>
          <Square size={13} /> Stop after this task
        </Button>
        <Button variant="danger" disabled={stopping === 'now'} onClick={() => onStop('now')}>
          Stop now
        </Button>
      </div>
    </div>
  )
}

function Scoreboard({ state, onState }: { state: BatchState; onState: (s: BatchState) => void }) {
  const [openId, setOpenId] = useState<string | null>(null)
  const [confirmId, setConfirmId] = useState<string | null>(null)
  const remove = useCallback(
    async (id: string): Promise<void> => {
      try {
        onState(await invoke('batch:delete', id))
        setConfirmId(null)
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },
    [onState]
  )
  if (!state.runs.length) return <div className="text-[13px] text-muted">No batch runs yet. A finished run is scored here.</div>
  return (
    <div className="space-y-2">
      <div className="text-[13px] font-medium">Scoreboard</div>
      <div className="divide-y divide-border rounded-lg border border-border">
        {state.runs.map((run) => {
          const s = summarize(run)
          const isActive = run.id === state.activeId
          const expanded = openId === run.id
          return (
            <div key={run.id}>
              <div className="flex items-center gap-2 px-3 py-2">
                <button
                  className="flex min-w-0 flex-1 items-center gap-2 text-left"
                  aria-expanded={expanded}
                  onClick={() => setOpenId(expanded ? null : run.id)}
                >
                  {expanded ? <ChevronDown size={14} className="shrink-0 text-muted" /> : <ChevronRight size={14} className="shrink-0 text-muted" />}
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 text-[13px]">
                      <span className="truncate font-medium">{run.label || 'Unlabelled'}</span>
                      <Badge tone={run.status === 'running' ? 'accent' : run.status === 'done' ? 'ok' : 'neutral'}>{run.status}</Badge>
                      {run.rentalOk && <Badge tone="warn">{run.modelCloud ? 'OpenRouter' : 'rental'}</Badge>}
                    </div>
                    <div className="truncate text-[11.5px] text-muted">
                      {run.modelName} · {relativeTime(run.createdAt)} · {s.tasksSolved}/{s.tasksScored} tasks solved
                      {s.medianRounds !== null && <> · median {Math.round(s.medianRounds)} rounds</>}
                      {s.medianMs !== null && <> · {formatDuration(s.medianMs)}</>}
                    </div>
                  </div>
                  <div className="shrink-0 text-right tabular-nums">
                    <div className="text-[13px] font-medium">{pct(s.rate)}</div>
                    <div className="text-[11.5px] text-muted">
                      {s.solved}/{s.scored}
                    </div>
                  </div>
                </button>
                {confirmId === run.id ? (
                  <div className="flex shrink-0 items-center gap-1">
                    <Button size="sm" variant="danger" onClick={() => void remove(run.id)}>
                      Remove
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => setConfirmId(null)}>
                      Keep
                    </Button>
                  </div>
                ) : (
                  <IconButton label={isActive ? 'Running now' : 'Remove run'} disabled={isActive} onClick={() => setConfirmId(run.id)}>
                    <Trash2 size={15} />
                  </IconButton>
                )}
              </div>
              {expanded && <RunDetail run={run} />}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function RunDetail({ run }: { run: BatchRun }) {
  const s = summarize(run)
  const counts = (Object.entries(s.counts) as [BatchItemStatus, number][]).filter(([, n]) => n > 0)
  return (
    <div className="space-y-3 border-t border-border bg-panel px-3 py-3">
      {run.stopNote && <div className="rounded-lg border border-warn/40 bg-warn/10 p-2.5 text-[12.5px]">{run.stopNote}</div>}
      <div className="flex flex-wrap gap-1.5">
        {counts.map(([status, n]) => (
          <Badge key={status} tone={STATUS_TONE[status]}>
            <span className="tabular-nums">{n}</span> {STATUS_LABEL[status]}
          </Badge>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
        {s.families.map((f) => (
          <div key={f.family} className="rounded-md border border-border px-2.5 py-1.5">
            <div className="truncate text-[12px] font-medium">{f.family}</div>
            <div className="text-[11.5px] tabular-nums text-muted">
              {f.score.solved}/{f.score.scored} · {pct(f.score.rate)}
            </div>
          </div>
        ))}
      </div>
      <div className="divide-y divide-border rounded-md border border-border">
        {run.items.map((item, i) => (
          <ItemRow key={`${item.taskId}-${item.attempt}-${i}`} item={item} />
        ))}
      </div>
    </div>
  )
}

function ItemRow({ item }: { item: BatchItem }) {
  const wrong = item.status === 'wrong' || item.status === 'no-answer'
  return (
    <div className="px-2.5 py-1.5">
      <div className="flex items-center gap-2 text-[12px]">
        <span className="font-mono">{item.taskId}</span>
        <span className="text-faint tabular-nums">try {item.attempt}</span>
        <Badge tone={STATUS_TONE[item.status]} className="h-5 px-2 text-[11px]">
          {STATUS_LABEL[item.status]}
        </Badge>
        <span className="ml-auto text-[11.5px] tabular-nums text-muted">
          {item.toolRounds !== undefined && <>{item.toolRounds} rounds · </>}
          {item.ms !== undefined ? formatDuration(item.ms) : ''}
        </span>
      </div>
      {wrong && (
        <div className="text-[11.5px] text-faint">
          answered {item.answer ?? '—'} · expected {item.expected ?? '—'}
        </div>
      )}
      {item.error && <div className="text-[11.5px] text-danger">{item.error}</div>}
    </div>
  )
}
