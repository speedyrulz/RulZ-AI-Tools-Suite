import { ChevronRight, ListChecks, LoaderCircle, MessageSquare, Play, SkipForward, Square, Trash2 } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import {
  DEFAULT_BATCH_OPTIONS,
  isFinished,
  planBatch,
  summarize,
  type BatchItem,
  type BatchItemStatus,
  type BatchOptions,
  type BatchRun,
  type BatchScore
} from '@shared/batch'
import { starterToolsThatAsk } from '@shared/re'
import { DEFAULT_BATCH_TARGET, type BatchTarget } from '@shared/remote'
import type { RePracticeTask } from '@shared/types'
import { batchAdapterLabel } from '../../lib/adapters'
import { cn, formatCompact, formatDuration, formatTokens, relativeTime } from '../../lib/format'
import { batchModelId, costPerSolved, money, resumeTarget } from '../../lib/remote'
import { useApp } from '../../stores/app'
import { useModels } from '../../stores/models'
import { useNow } from '../remote/useNow'
import { useHostChoices, useTargetState } from '../remote/TargetPicker'
import { useBatch } from '../../stores/batch'
import { practiceRecords, solvedTasks, useReTraces } from '../../stores/reTraces'
import { isChatBusy, useChat } from '../../stores/chat'
import { availableTools, useMcp } from '../../stores/mcp'
import { Badge, Button, Dialog, IconButton, Input, NumberInput, ProgressBar, Switch, Tooltip } from '../ui'

const STATUS: Record<BatchItemStatus, { label: string; tone: 'neutral' | 'accent' | 'warn' | 'danger' | 'ok'; hint: string }> = {
  pending: { label: 'Not run', tone: 'neutral', hint: 'Not tried yet.' },
  running: { label: 'Running', tone: 'accent', hint: 'In progress.' },
  solved: { label: 'Solved', tone: 'ok', hint: 'The ANSWER line passed the task’s check.' },
  wrong: { label: 'Wrong', tone: 'danger', hint: 'An ANSWER line that did not pass the check.' },
  'no-answer': { label: 'No answer', tone: 'danger', hint: 'The final reply had no ANSWER line.' },
  rounds: { label: 'Round limit', tone: 'warn', hint: 'Still calling tools at the tool round limit.' },
  timeout: { label: 'Time limit', tone: 'warn', hint: 'Still working at the time limit.' },
  error: { label: 'Error', tone: 'neutral', hint: 'It could not run or be graded. Not counted in the score.' },
  stopped: { label: 'Stopped', tone: 'neutral', hint: 'You stopped this session in the chat. Not counted in the score.' }
}

const pct = (s: BatchScore): string => (s.rate === null ? '—' : `${Math.round(s.rate * 100)}%`)

const baseName = (p: string): string => p.split(/[\\/]/).filter(Boolean).at(-1) ?? p

/** Options and the Start button for a batch run, and its progress while one runs. */
export function BatchStart({ tasks, family, kitDir }: { tasks: RePracticeTask[]; family: string; kitDir: string }) {
  const summaries = useChat((s) => s.summaries)
  const chatBusy = useChat((s) => isChatBusy(s))
  const runs = useBatch((s) => s.runs)
  const activeId = useBatch((s) => s.activeId)
  const starting = useBatch((s) => s.starting)
  const start = useBatch((s) => s.start)
  const init = useBatch((s) => s.init)
  const settings = useApp((s) => s.settings)
  const update = useApp((s) => s.updateSettings)
  const servers = useMcp((s) => s.servers)
  // Starter tools that would be refused in a run nobody watches. `servers` is what availableTools reads.
  const asking = useMemo(() => (settings ? starterToolsThatAsk(settings, availableTools()) : []), [settings, servers])
  const [opts, setOpts] = useState<BatchOptions>(DEFAULT_BATCH_OPTIONS)
  const [label, setLabel] = useState('')
  useEffect(() => void init(), [init])
  // The run's model decides which hosts are valid: the batch store takes the RE model if the list has it, else the active one.
  const models = useModels((s) => s.models)
  const activeModelId = useModels((s) => s.activeModelId)
  const runModelId = batchModelId(settings?.re.modelId, models, activeModelId)
  const where = useTargetState(runModelId)

  const options = useMemo(() => ({ ...opts, family }), [opts, family])
  const traces = useReTraces((s) => s.traces)
  useEffect(() => useReTraces.getState().init(), [])
  const planned = useMemo(() => planBatch(tasks, options, solvedTasks(practiceRecords(summaries, traces))), [tasks, options, summaries, traces])
  const taskCount = options.attempts > 0 ? planned.length / options.attempts : 0
  const set = <K extends keyof BatchOptions>(key: K, value: BatchOptions[K]): void => setOpts((o) => ({ ...o, [key]: value }))

  const active = runs.find((r) => r.id === activeId)
  if (active) return <ActiveRun run={active} />

  return (
    <div className="rounded-lg border border-border bg-panel-2/50 p-3">
      <div className="mb-1 text-[13px] font-medium">Batch run</div>
      <div className="mb-2 text-[11.5px] text-muted">
        Runs the tasks without you, one or several at once, on this PC or on rented vast.ai instances. It grades each and keeps a score for the
        model. Tools that would ask for approval are refused instead of waiting. Each task is a normal recorded session, so the solved ones can
        go into training.
      </div>
      {asking.length > 0 && settings && (
        <div className="mb-2 flex items-center gap-3 rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">
          <div className="min-w-0 flex-1">
            {asking.length === 1 ? 'A tool every session starts with asks' : 'Tools every session starts with ask'} for approval, so a batch run would
            refuse {asking.length === 1 ? 'it' : 'them'}: <span className="font-mono">{asking.join(', ')}</span>
          </div>
          <Button size="sm" onClick={() => void update({ allowedTools: [...new Set([...settings.allowedTools, ...asking])] })}>
            Always allow
          </Button>
        </div>
      )}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-[12.5px]">
        <span className="flex items-center gap-1.5">
          At most
          <NumberInput className="w-16" value={opts.maxTasks || null} placeholder="all" min={0} onChange={(v) => set('maxTasks', Math.max(0, v ?? 0))} />
          tasks
        </span>
        <span className="flex items-center gap-1.5">
          Tries per task
          <NumberInput className="w-14" value={opts.attempts} min={1} max={10} onChange={(v) => set('attempts', Math.min(10, Math.max(1, v ?? 1)))} />
        </span>
        <Tooltip content="A task still working this long after its first reply started is stopped and counted as not solved. Loading the model does not count. Empty = no limit.">
          <span className="flex items-center gap-1.5">
            Minutes per task
            <NumberInput className="w-16" value={opts.minutesPerTask || null} placeholder="none" min={0} max={1440} onChange={(v) => set('minutesPerTask', Math.min(1440, Math.max(0, v ?? 0)))} />
          </span>
        </Tooltip>
        <Tooltip content="Leaves out tasks an earlier session solved. Good for collecting training data; leave it off to compare models on the same tasks.">
          <label className="flex items-center gap-1.5">
            <Switch checked={opts.onlyUnsolved} onCheckedChange={(v) => set('onlyUnsolved', v)} /> Only unsolved
          </label>
        </Tooltip>
      </div>
      <div className="mt-3 border-t border-border/60 pt-3">{where.view}</div>
      <div className="mt-3 flex items-center gap-3">
        <Input className="h-7.5 max-w-64 text-xs" value={label} maxLength={80} placeholder="Label, e.g. base or lora-v2 (optional)" onChange={(e) => setLabel(e.target.value)} />
        <div className="flex-1 text-right text-[11.5px] text-faint tabular-nums">
          {planned.length === 0
            ? 'No tasks in this selection'
            : `${taskCount} task${taskCount === 1 ? '' : 's'}${options.attempts > 1 ? ` × ${options.attempts} tries = ${planned.length} sessions` : ''}${family ? ` · only ${family}` : ''}`}
        </div>
        <Button variant="primary" disabled={!planned.length || chatBusy || starting || where.problems.length > 0} onClick={() => void start(tasks, options, label, kitDir, where.target)}>
          {starting ? <LoaderCircle size={14} className="animate-spin" /> : <Play size={14} />} Start batch run
        </Button>
      </div>
    </div>
  )
}

/** The run in progress: every running task with its host and minutes, with Open and Skip; stop controls. */
function ActiveRun({ run: active }: { run: BatchRun }) {
  const stopping = useBatch((s) => s.stopping)
  const stop = useBatch((s) => s.stop)
  const skip = useBatch((s) => s.skip)
  const setPage = useApp((s) => s.setPage)
  const open = useChat((s) => s.open)
  const now = useNow(5000)
  const s = summarize(active)
  const running = active.items.filter((i) => i.status === 'running')
  const t = active.target
  return (
    <div className="rounded-lg border border-accent/40 bg-accent-soft/40 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2 text-[13px] font-medium">
          <LoaderCircle size={14} className="shrink-0 animate-spin text-accent" />
          <span className="truncate">
            Batch run{active.label ? ` “${active.label}”` : ''}: {s.finished} of {s.total} tasks done
            {running.length > 1 ? `, ${running.length} running at once` : ''}
          </span>
        </div>
        <div className="flex shrink-0 gap-1.5">
          <Tooltip content="Lets the tasks in progress finish, then stops. The rest can be resumed later.">
            <Button size="sm" disabled={!!stopping} onClick={() => stop('after')}>
              Stop after these tasks
            </Button>
          </Tooltip>
          <Tooltip content="Stops the tasks in progress too. They are not counted, and resuming the run starts them again.">
            <Button size="sm" variant="danger" disabled={stopping === 'now'} onClick={() => stop('now')}>
              <Square size={12} /> Stop now
            </Button>
          </Tooltip>
        </div>
      </div>
      <ProgressBar value={s.total ? s.finished / s.total : 0} />
      <ul className="mt-2 space-y-1">
        {running.length === 0 && <li className="text-[11.5px] text-muted">Starting the next tasks…</li>}
        {running.map((i) => (
          <li key={`${i.taskId}-${i.attempt}`} className="flex items-center gap-2 text-[11.5px] tabular-nums">
            <span className="min-w-0 flex-1 truncate">
              <span className="font-mono">{i.taskId}</span>
              {active.options.attempts > 1 ? ` (try ${i.attempt})` : ''}
              <span className="text-faint"> · {i.host ?? 'This PC'}</span>
            </span>
            <span className="shrink-0 text-muted">{i.startedAt ? `${Math.max(0, Math.floor((now - i.startedAt) / 60000))} min` : 'starting'}</span>
            {i.conversationId && (
              <Button
                size="sm"
                variant="ghost"
                onClick={async () => {
                  await open(i.conversationId!)
                  setPage('chat')
                }}
              >
                <MessageSquare size={13} /> Open
              </Button>
            )}
            {i.conversationId && (
              <Tooltip content="Stops this task. It counts as stopped, and the run takes the next one.">
                <Button size="sm" variant="ghost" onClick={() => skip(i.conversationId!)}>
                  <SkipForward size={13} /> Skip
                </Button>
              </Tooltip>
            )}
          </li>
        ))}
      </ul>
      <div className="mt-1.5 flex flex-wrap gap-x-3 text-[11.5px] text-muted tabular-nums">
        <span>
          {s.solved} of {s.scored} solved
        </span>
        {t && (
          <span>
            {t.sessionsPerHost} at once on each of {t.hosts.length} host{t.hosts.length === 1 ? '' : 's'}
          </span>
        )}
        {active.cost !== undefined && <span>about {money(active.cost)} so far</span>}
        {t?.stopRentalsAfter && <span>Rentals stop when it ends</span>}
        {stopping && <span className="text-warn">{stopping === 'now' ? 'Stopping…' : 'Stopping after these tasks…'}</span>}
      </div>
    </div>
  )
}

function ItemRows({ run }: { run: BatchRun }) {
  const open = useChat((s) => s.open)
  const setPage = useApp((s) => s.setPage)
  const exists = useChat((s) => s.summaries)
  const known = useMemo(() => new Set(exists.map((s) => s.id)), [exists])
  const multi = run.options.attempts > 1
  return (
    <div className="max-h-80 overflow-y-auto rounded-lg border border-border">
      <table className="w-full table-fixed text-[12px]">
        <thead className="sticky top-0 bg-panel-2 text-[10.5px] tracking-wide text-faint uppercase">
          <tr>
            <th className="px-3 py-1.5 text-left font-medium">Task</th>
            <th className="w-48 px-2 py-1.5 text-left font-medium">Result</th>
            <th className="w-18 px-2 py-1.5 text-right font-medium">Rounds</th>
            <th className="w-20 px-2 py-1.5 text-right font-medium">Time</th>
            <th className="w-20 px-2 py-1.5 text-right font-medium">
              <Tooltip content="The largest prompt of the session, in tokens: how full the context got.">
                <span>Context</span>
              </Tooltip>
            </th>
            <th className="w-10" />
          </tr>
        </thead>
        <tbody>
          {run.items.map((i: BatchItem, n) => {
            const st = STATUS[i.status]
            return (
              <tr key={n} className="border-t border-border/60 hover:bg-hover/40">
                <td className="max-w-0 px-3 py-1">
                  <div className="truncate font-mono text-[11.5px]">
                    {i.taskId}
                    {multi && <span className="text-faint"> · try {i.attempt}</span>}
                  </div>
                  <div className="truncate text-[10.5px] text-faint">
                    {i.family} · {i.tier}
                    {i.host ? ` · ${i.host}` : ''}
                  </div>
                </td>
                <td className="px-2 py-1">
                  <Tooltip content={i.error ?? (i.answer ? `${st.hint} Answer given: ${i.answer}` : st.hint)}>
                    <span className="inline-flex items-center gap-1">
                      <Badge tone={st.tone}>{st.label}</Badge>
                      {!!i.denied && <Badge tone="warn">{i.denied} refused</Badge>}
                    </span>
                  </Tooltip>
                </td>
                <td className="px-2 py-1 text-right text-muted tabular-nums">{i.toolRounds ?? '—'}</td>
                <td className="px-2 py-1 text-right whitespace-nowrap text-muted tabular-nums">{isFinished(i.status) ? formatDuration(i.ms) : '—'}</td>
                <td className="px-2 py-1 text-right text-muted tabular-nums">{i.promptTokens ? formatCompact(i.promptTokens) : '—'}</td>
                <td className="px-2 py-1">
                  {i.conversationId && known.has(i.conversationId) && (
                    <IconButton
                      size="sm"
                      label="Open in chat"
                      onClick={async () => {
                        await open(i.conversationId!)
                        setPage('chat')
                      }}
                    >
                      <MessageSquare size={13} />
                    </IconButton>
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function RunRow({ run }: { run: BatchRun }) {
  const activeId = useBatch((s) => s.activeId)
  const starting = useBatch((s) => s.starting)
  const resume = useBatch((s) => s.resume)
  const remove = useBatch((s) => s.remove)
  const chatBusy = useChat((s) => isChatBusy(s))
  const [open, setOpen] = useState(false)
  const [asking, setAsking] = useState(false)
  // Where it ran before is used again when every host of it can still run the model; otherwise the user chooses.
  const choices = useHostChoices(run.modelId)
  const again = resumeTarget(run.target ?? DEFAULT_BATCH_TARGET, choices)
  const s = useMemo(() => summarize(run), [run])
  const running = activeId === run.id
  const left = s.counts.pending
  const o = run.options
  return (
    <>
      <tr className="border-t border-border/60 hover:bg-hover/40">
        <td className="max-w-0 px-3 py-1.5">
          <button className="flex w-full items-center gap-1.5 text-left" onClick={() => setOpen(!open)} aria-expanded={open}>
            <ChevronRight size={13} className={cn('shrink-0 text-faint transition-transform', open && 'rotate-90')} />
            <span className="min-w-0">
              <span className="flex items-center gap-1.5">
                <span className="truncate">{run.label || baseName(run.modelName || run.modelId || 'Unknown model')}</span>
                {running ? <Badge tone="accent">running</Badge> : run.status === 'stopped' && <Badge tone="warn">stopped</Badge>}
              </span>
              <span className="block truncate text-[10.5px] text-faint" title={run.modelId ?? ''}>
                {run.label ? `${baseName(run.modelName || run.modelId || '')} · ` : ''}
                {run.adapters?.length ? `${batchAdapterLabel(run.adapters)} · ` : ''}
                {run.contextLength ? `${formatTokens(run.contextLength)} context · ` : ''}
                {run.hosts?.length ? `${run.hosts.map((h) => h.label).join(', ')} · ` : ''}
                {run.target && run.target.sessionsPerHost > 1 ? `${run.target.sessionsPerHost} at once · ` : ''}
                {relativeTime(run.createdAt)}
              </span>
            </span>
          </button>
        </td>
        <td className="px-2 py-1.5 text-right whitespace-nowrap tabular-nums">
          <Tooltip
            content={`${s.solved} of ${s.scored} tries solved. Errors and stopped tasks are not counted.${
              o.attempts > 1 ? ` ${s.tasksSolved} of ${s.tasksScored} tasks solved in at least one try.` : ''
            }`}
          >
            <span>
              <span className="font-medium">{pct(s)}</span> <span className="text-faint">{s.solved}/{s.scored}</span>
            </span>
          </Tooltip>
        </td>
        <td className="px-2 py-1.5 text-right whitespace-nowrap text-muted tabular-nums">
          {s.finished}/{s.total}
        </td>
        <td className="px-2 py-1.5 text-right text-muted tabular-nums">{s.medianRounds === null ? '—' : Math.round(s.medianRounds)}</td>
        <td className="px-2 py-1.5 text-right whitespace-nowrap text-muted tabular-nums">{formatDuration(s.medianMs)}</td>
        <td className="px-2 py-1.5 text-right whitespace-nowrap text-muted tabular-nums">
          {run.cost === undefined ? (
            '—'
          ) : (
            <Tooltip content="Estimated rental cost of the run, and the cost for each task it solved.">
              <span>
                {money(run.cost)}
                {costPerSolved(run.cost, s.solved) !== null && <span className="block text-[10.5px] text-faint">{money(costPerSolved(run.cost, s.solved))} per solved</span>}
              </span>
            </Tooltip>
          )}
        </td>
        <td className="px-2 py-1.5">
          <div className="flex justify-end gap-0.5">
            {!running && left > 0 && (
              <IconButton size="sm" label={`Run the ${left} task${left === 1 ? '' : 's'} not run yet`} disabled={!!activeId || starting || chatBusy} onClick={() => (again ? void resume(run.id, again) : setAsking(true))}>
                <Play size={13} />
              </IconButton>
            )}
            <IconButton size="sm" label="Remove this run from the list (its chats stay)" disabled={running || starting} onClick={() => void remove(run.id)}>
              <Trash2 size={13} />
            </IconButton>
          </div>
        </td>
      </tr>
      {asking && <ResumeDialog run={run} onClose={() => setAsking(false)} />}
      {open && (
        <tr className="border-t border-border/60">
          <td colSpan={7} className="space-y-2 bg-panel-2/40 px-3 py-2.5">
            <div className="flex flex-wrap gap-1.5 text-[11.5px]">
              {s.families.map((f) => (
                <span key={f.family} className="rounded-md border border-border px-2 py-0.5 tabular-nums">
                  {f.family} <span className="font-medium">{pct(f.score)}</span>{' '}
                  <span className="text-faint">
                    {f.score.solved}/{f.score.scored}
                  </span>
                </span>
              ))}
            </div>
            <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-faint tabular-nums">
              {(['wrong', 'no-answer', 'rounds', 'timeout', 'error', 'stopped'] as const)
                .filter((k) => s.counts[k] > 0)
                .map((k) => (
                  <span key={k}>
                    {STATUS[k].label}: {s.counts[k]}
                  </span>
                ))}
              <span>
                Limits: {run.maxToolRounds} tool rounds, {o.minutesPerTask ? `${o.minutesPerTask} min per task` : 'no time limit'}
              </span>
              {o.attempts > 1 && <span>{o.attempts} tries per task</span>}
              {run.target && (
                <span>
                  {run.target.sessionsPerHost} task{run.target.sessionsPerHost === 1 ? '' : 's'} at once on each of {run.target.hosts.length} host
                  {run.target.hosts.length === 1 ? '' : 's'}
                </span>
              )}
              {o.family && <span>Only {o.family}</span>}
              {o.onlyUnsolved && <span>Only unsolved</span>}
              {s.outputTokens > 0 && <span>{formatCompact(s.outputTokens)} tokens written</span>}
              {s.denied > 0 && <span className="text-warn">{s.denied} tool calls refused (needed approval)</span>}
            </div>
            <ItemRows run={run} />
          </td>
        </tr>
      )}
    </>
  )
}

/** Asks where to run the rest of a stopped run when where it ran before cannot run it now (a rental stopped, a model not set up). */
function ResumeDialog({ run, onClose }: { run: BatchRun; onClose: () => void }) {
  const resume = useBatch((s) => s.resume)
  const where = useTargetState(run.modelId, run.target as BatchTarget | undefined)
  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title="Where should the rest run?"
      description="Where this run ran before cannot run it now. Choose again."
      className="w-[min(640px,calc(100vw-32px))]"
      footer={
        <>
          <Button autoFocus variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            disabled={where.problems.length > 0}
            onClick={() => {
              onClose()
              void resume(run.id, where.target)
            }}
          >
            <Play size={13} /> Run the rest
          </Button>
        </>
      }
    >
      <div className="px-5 py-4">{where.view}</div>
    </Dialog>
  )
}

/** The scoreboard: every batch run with its score. */
export function BatchResults() {
  const runs = useBatch((s) => s.runs)
  const loaded = useBatch((s) => s.loaded)
  const init = useBatch((s) => s.init)
  useEffect(() => void init(), [init])
  if (!loaded || runs.length === 0) return null
  return (
    <section className="rounded-xl border border-border bg-panel">
      <div className="flex items-center gap-1.5 border-b border-border px-4 py-2.5 text-[11px] font-semibold tracking-wide text-faint uppercase">
        <ListChecks size={12} /> Batch runs
      </div>
      <div className="space-y-3 p-4">
        <div className="text-xs text-muted">
          One row per batch run. Compare runs that used the same tasks and limits: the score is solved tries out of the tries that ran to an
          end.
        </div>
        <div className="max-h-[32rem] overflow-y-auto rounded-lg border border-border">
          <table className="w-full table-fixed text-[12px]">
            <thead className="sticky top-0 z-10 bg-panel-2 text-[10.5px] tracking-wide text-faint uppercase">
              <tr>
                <th className="px-3 py-1.5 text-left font-medium">Run</th>
                <th className="w-24 px-2 py-1.5 text-right font-medium">Solved</th>
                <th className="w-16 px-2 py-1.5 text-right font-medium">
                  <Tooltip content="Tasks that have been run, of the tasks in the run.">
                    <span>Done</span>
                  </Tooltip>
                </th>
                <th className="w-20 px-2 py-1.5 text-right font-medium">
                  <Tooltip content="Median tool rounds of the solved tries.">
                    <span>Rounds</span>
                  </Tooltip>
                </th>
                <th className="w-20 px-2 py-1.5 text-right font-medium">
                  <Tooltip content="Median time of the solved tries.">
                    <span>Time</span>
                  </Tooltip>
                </th>
                <th className="w-24 px-2 py-1.5 text-right font-medium">
                  <Tooltip content="Estimated rental cost of the run, and per solved task. Empty for a run on this PC.">
                    <span>Cost</span>
                  </Tooltip>
                </th>
                <th className="w-18" />
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <RunRow key={r.id} run={r} />
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}
