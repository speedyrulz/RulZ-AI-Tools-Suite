import { Folder, Play, RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { familyCounts, nextPracticeTask } from '@shared/batch'
import { practiceScope, triesText } from '@shared/modes'
import type { BatchState, ReKind, ReOutcome, ReOverview, ReStartRequest } from '@shared/types'
import { toast } from 'sonner'
import { invoke } from '../lib/api'
import { cn, relativeTime } from '../lib/format'
import { activeModelId, useApp } from '../stores/app'
import { useWorkspace } from '../stores/workspace'
import { modelLabel } from './common'
import { FolderPicker } from './FolderPicker'
import { BatchRunBox, BatchScoreboard, friendlyReError, useBatchState } from './PracticeBatch'
import { ReloadBar } from './ReloadBar'
import { Badge, Button, Field, Input, Select, Sheet, Spinner, Switch, Textarea } from './ui'

const OUTCOME_TONE: Record<ReOutcome, 'neutral' | 'ok' | 'warn' | 'danger'> = { unrated: 'neutral', solved: 'ok', partial: 'warn', failed: 'danger' }
const KIND_LABEL: Record<ReKind, string> = { analyse: 'Analyse', practice: 'Practice', 'clone-pieces': 'Clone pieces', 'clone-full': 'Full clone' }

/** A computer running an older RulZ Coding Agent doesn't know the RE or batch requests. */
const friendly = friendlyReError

/**
 * RulZ LLM Suite's RE overview, through the computer, for the model in use (the practice counts are
 * that model's; an older computer or RulZ LLM Suite ignores the model and counts all models together).
 * Asked again when the model changes; an answer for a model picked before is dropped.
 */
export function useReOverview(open: boolean, model?: string): { data: ReOverview | null; error: string | null; loading: boolean; reload: () => void } {
  const [data, setData] = useState<ReOverview | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  /** The newest question. */
  const latest = useRef(0)
  const reload = useCallback(() => {
    const mine = ++latest.current
    setLoading(true)
    invoke('re:overview', model || undefined)
      .then((d) => {
        if (mine !== latest.current) return
        setData(d)
        setError(null)
      })
      .catch((e) => mine === latest.current && setError(friendly(e)))
      .finally(() => mine === latest.current && setLoading(false))
  }, [model])
  useEffect(() => {
    if (open) reload()
  }, [open, reload])
  return { data, error, loading, reload }
}

export type ReTab = 'start' | 'practice' | 'sessions'

/** Reverse engineering from the phone: the tasks RulZ LLM Suite's RE page offers, and what was done. */
export function ReSheet({ open, onOpenChange, initialTab = 'start' }: { open: boolean; onOpenChange: (open: boolean) => void; initialTab?: ReTab }) {
  const project = useWorkspace((s) => s.project)
  const running = useWorkspace((s) => s.running)
  const modelId = useApp((s) => activeModelId(s.settings, s.connection))
  const { data, error, loading, reload } = useReOverview(open, modelId)
  const { state: batch, supported: batchSupported, setState: setBatch } = useBatchState(open, reload)
  const batchActive = Boolean(batch?.activeId)
  const busy = running || batchActive
  const [tab, setTab] = useState<ReTab>(initialTab)
  const [record, setRecord] = useState(true)
  useEffect(() => {
    if (open) setTab(initialTab)
  }, [open, initialTab])

  const start = async (req: ReStartRequest): Promise<void> => {
    if (!project) return
    try {
      await invoke('re:start', project.path, { ...req, record })
      onOpenChange(false)
    } catch (e) {
      toast.error(friendly(e))
    }
  }

  return (
    <Sheet
      open={open}
      onOpenChange={onOpenChange}
      title="Reverse engineering"
      full
      footer={
        <label className="flex items-center gap-3 text-[13px] text-muted">
          <Switch checked={record} onCheckedChange={setRecord} />
          Record for training in RulZ LLM Suite
        </label>
      }
    >
      <div className="flex items-center gap-1 px-4 pb-2">
        {(
          [
            ['start', 'Start'],
            ['practice', `Practice${data ? ` ${data.practice.tasks.length}` : ''}`],
            ['sessions', `Done${data ? ` ${data.sessions.length}` : ''}`]
          ] as const
        ).map(([value, label]) => (
          <button
            key={value}
            onClick={() => setTab(value)}
            className={cn('rounded-lg px-3 py-1.5 text-[14px]', tab === value ? 'bg-accent-soft text-accent' : 'text-muted active:bg-hover')}
          >
            {label}
          </button>
        ))}
        <div className="flex-1" />
        <button className="rounded-lg p-2 text-muted active:bg-hover" aria-label="Refresh" onClick={reload}>
          {loading ? <Spinner /> : <RefreshCw size={17} />}
        </button>
      </div>
      <div className="px-4 pb-4">
        {/* The next task here is a reverse-engineering one, so the Reverse engineering settings are what it should run with. */}
        {open && <ReloadBar mode="re" busy={batchActive} className="mb-3" />}
        {batchActive ? (
          <div className="mb-3 text-[13px] text-warn">A batch run is going; stop it before starting another task.</div>
        ) : (
          running && <div className="mb-3 text-[13px] text-warn">A task is running; wait for it to finish before starting another.</div>
        )}
        {error ? (
          <div className="rounded-xl border border-warn/40 bg-warn/10 p-3 text-[14px]">{error}</div>
        ) : !data ? (
          <div className="flex items-center gap-2 text-[14px] text-muted">
            <Spinner /> Asking RulZ LLM Suite…
          </div>
        ) : tab === 'start' ? (
          <StartForm data={data} busy={busy} onStart={start} />
        ) : tab === 'practice' ? (
          <PracticeList
            data={data}
            busy={busy}
            record={record}
            batch={batch}
            batchSupported={batchSupported}
            onBatchChange={setBatch}
            onReload={reload}
            onStart={(taskId) => void start({ kind: 'practice', taskId })}
          />
        ) : (
          <SessionList data={data} />
        )}
      </div>
    </Sheet>
  )
}

function Chips({ items, value, onPick }: { items: { label: string; text: string }[]; value: string; onPick: (t: string) => void }) {
  return (
    <div className="mb-2 flex flex-wrap gap-1.5">
      {items.map((g) => (
        <button
          key={g.label}
          onClick={() => onPick(g.text)}
          className={cn('rounded-lg border px-2.5 py-1 text-[13px]', value === g.text ? 'border-accent/50 bg-accent-soft text-accent' : 'border-border text-muted active:bg-hover')}
        >
          {g.label}
        </button>
      ))}
    </div>
  )
}

function StartForm({ data, busy, onStart }: { data: ReOverview; busy: boolean; onStart: (req: ReStartRequest) => Promise<void> }) {
  const project = useWorkspace((s) => s.project)
  const [mode, setMode] = useState<'analyse' | 'clone'>('analyse')
  const [target, setTarget] = useState('')
  const [browse, setBrowse] = useState(false)
  const [goal, setGoal] = useState(data.goals[0]?.text ?? '')
  const [scope, setScope] = useState<'pieces' | 'full'>('pieces')
  const [pieces, setPieces] = useState(data.clonePieces[0]?.text ?? '')
  const [notes, setNotes] = useState('')
  const [language, setLanguage] = useState('')
  const [starting, setStarting] = useState(false)
  const ready = !!target.trim() && (mode === 'analyse' ? !!goal.trim() : scope === 'full' || !!pieces.trim())
  const go = async (): Promise<void> => {
    setStarting(true)
    await onStart(
      mode === 'analyse'
        ? { kind: 'analyse', target: target.trim(), goal }
        : { kind: 'clone', target: target.trim(), scope, details: scope === 'full' ? notes : pieces, language }
    )
    setStarting(false)
  }
  return (
    <div className="space-y-4">
      <div className="flex gap-1 rounded-xl border border-border p-1 text-[14px]">
        {(
          [
            ['analyse', 'Analyse'],
            ['clone', 'Clone']
          ] as const
        ).map(([v, l]) => (
          <button key={v} onClick={() => setMode(v)} className={cn('flex-1 rounded-lg py-1.5', mode === v ? 'bg-accent-soft text-accent' : 'text-muted')}>
            {l}
          </button>
        ))}
      </div>
      <Field label="Program or folder on the computer">
        <div className="flex gap-2">
          <Input className="font-mono text-[13px]" value={target} placeholder="C:\path\to\program.exe" onChange={(e) => setTarget(e.target.value)} />
          <Button aria-label="Browse folders" onClick={() => setBrowse(true)}>
            <Folder size={16} />
          </Button>
        </div>
      </Field>
      <FolderPicker
        open={browse}
        onOpenChange={setBrowse}
        title="Choose the app's folder"
        start={target.trim() || project?.path || ''}
        action="Use this folder"
        onPick={(p) => setTarget(p)}
      />
      {mode === 'analyse' ? (
        <Field label="What to find out">
          <Chips items={data.goals} value={goal} onPick={setGoal} />
          <Textarea rows={3} value={goal} onChange={(e) => setGoal(e.target.value)} />
        </Field>
      ) : (
        <>
          <Field label="What to clone">
            <div className="mb-2 flex gap-5 text-[14px]">
              {(
                [
                  ['pieces', 'Some pieces'],
                  ['full', 'All of it']
                ] as const
              ).map(([v, l]) => (
                <label key={v} className="flex items-center gap-2">
                  <input type="radio" name="re-clone-scope" checked={scope === v} onChange={() => setScope(v)} /> {l}
                </label>
              ))}
            </div>
            {scope === 'pieces' ? (
              <>
                <Chips items={data.clonePieces} value={pieces} onPick={setPieces} />
                <Textarea rows={3} value={pieces} onChange={(e) => setPieces(e.target.value)} />
              </>
            ) : (
              <Textarea rows={2} value={notes} placeholder="Optional: anything to keep, change or leave out" onChange={(e) => setNotes(e.target.value)} />
            )}
          </Field>
          <Field label="Language" hint="The new code goes in this project.">
            <Select value={language} onChange={(e) => setLanguage(e.target.value)}>
              {data.cloneLanguages.map((l) => (
                <option key={l} value={l}>
                  {l || 'Same as the original'}
                </option>
              ))}
            </Select>
          </Field>
        </>
      )}
      <Button variant="primary" size="lg" className="w-full" disabled={!ready || busy || starting} onClick={() => void go()}>
        <Play size={16} /> {mode === 'clone' ? 'Start cloning' : 'Start'}
      </Button>
      <p className="text-[12px] text-faint">Runs on the computer with the REA tools set up in RulZ LLM Suite. Only work on software you have the right to.</p>
    </div>
  )
}

function PracticeList({
  data,
  busy,
  record,
  batch,
  batchSupported,
  onBatchChange,
  onReload,
  onStart
}: {
  data: ReOverview
  busy: boolean
  record: boolean
  batch: BatchState | null
  batchSupported: boolean
  onBatchChange: (s: BatchState) => void
  onReload: () => void
  onStart: (taskId: string) => void
}) {
  const project = useWorkspace((s) => s.project)
  const [family, setFamily] = useState('')
  const [unsolved, setUnsolved] = useState(false)
  const [kitDraft, setKitDraft] = useState(data.practice.kitDir)
  const [browseKit, setBrowseKit] = useState(false)
  const [savingKit, setSavingKit] = useState(false)

  useEffect(() => {
    setKitDraft(data.practice.kitDir)
  }, [data.practice.kitDir])

  const allTasks = data.practice.tasks
  const familyTiles = useMemo(() => familyCounts(allTasks), [allTasks])
  const families = useMemo(() => familyTiles.map((f) => f.family), [familyTiles])
  const scopedTasks = useMemo(() => allTasks.filter((t) => !family || t.family === family), [allTasks, family])
  const nextTask = useMemo(() => nextPracticeTask(scopedTasks), [scopedTasks])
  const tasks = useMemo(() => scopedTasks.filter((t) => !unsolved || t.solved === 0), [scopedTasks, unsolved])

  const saveKit = async (nextDir: string): Promise<void> => {
    const trimmed = nextDir.trim()
    if (!trimmed || trimmed === data.practice.kitDir) {
      setKitDraft(data.practice.kitDir)
      return
    }
    setSavingKit(true)
    try {
      await invoke('re:setKit', trimmed)
      onReload()
    } catch (e) {
      toast.error(friendly(e))
    } finally {
      setSavingKit(false)
    }
  }

  return (
    <div className="space-y-4">
      <Field label="Kit folder">
        <div className="flex gap-2">
          <Input
            className="font-mono text-[13px]"
            value={kitDraft}
            placeholder="C:\path\to\re-training-kit"
            disabled={savingKit}
            onChange={(e) => setKitDraft(e.target.value)}
            onBlur={() => void saveKit(kitDraft)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') e.currentTarget.blur()
            }}
          />
          <Button aria-label="Browse folders" disabled={savingKit} onClick={() => setBrowseKit(true)}>
            <Folder size={16} />
          </Button>
        </div>
      </Field>
      <FolderPicker
        open={browseKit}
        onOpenChange={setBrowseKit}
        title="Choose the practice kit folder"
        start={kitDraft.trim() || data.practice.kitDir || project?.path || ''}
        action="Use this folder"
        onPick={(p) => {
          setBrowseKit(false)
          setKitDraft(p)
          void saveKit(p)
        }}
      />

      {data.practice.error ? (
        <div className="rounded-xl border border-warn/40 bg-warn/10 p-3 text-[14px]">{data.practice.error}</div>
      ) : !allTasks.length ? (
        <div className="text-[14px] text-muted">No practice tasks in {data.practice.kitDir}.</div>
      ) : (
        <>
          {/* Which model the counts are for: one model with a RulZ LLM Suite that counts per model, else all of them together. */}
          <div className="text-[13px] font-medium text-muted">{practiceScope(data.practice, modelLabel)}</div>
          <div className="grid grid-cols-2 gap-2">
            {familyTiles.map((c) => (
              <button
                key={c.family}
                type="button"
                onClick={() => setFamily(family === c.family ? '' : c.family)}
                className={cn(
                  'rounded-xl border px-3 py-2 text-left transition-colors',
                  family === c.family ? 'border-accent/60 bg-accent-soft text-accent' : 'border-border bg-panel-2/40 active:bg-hover'
                )}
              >
                <div className="truncate text-[14px] font-medium">{c.family}</div>
                <div className="text-[12px] text-faint tabular-nums">
                  {c.solved}/{c.total} solved · {c.tried} tried{c.failed !== undefined && <> · {c.failed} failed</>}
                </div>
              </button>
            ))}
          </div>

          <div className="flex items-center justify-between gap-2">
            <span className="text-[13px] text-faint tabular-nums">
              {allTasks.length} tasks{family ? ` · only ${family}` : ''}
            </span>
            <Button variant="secondary" disabled={!nextTask || busy} onClick={() => nextTask && onStart(nextTask.id)}>
              <Play size={15} /> Start next task
            </Button>
          </div>

          {batchSupported && (
            <BatchRunBox
              tasks={allTasks}
              family={family}
              record={record}
              busy={busy}
              state={batch}
              onStateChange={onBatchChange}
            />
          )}
        </>
      )}

      {batchSupported && <BatchScoreboard state={batch} onStateChange={onBatchChange} />}

      {!data.practice.error && allTasks.length > 0 && (
        <div className="space-y-3 pt-1">
          <div className="text-[14px] font-medium">All tasks</div>
          <Select value={family} onChange={(e) => setFamily(e.target.value)}>
            <option value="">Every family</option>
            {families.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </Select>
          <label className="flex items-center gap-3 text-[14px]">
            <Switch checked={unsolved} onCheckedChange={setUnsolved} /> Only ones not solved yet
            <span className="ml-auto text-[12px] text-faint tabular-nums">{tasks.length} tasks</span>
          </label>
          <div className="divide-y divide-border rounded-xl border border-border">
            {tasks.map((t) => (
              <div key={t.id} className="flex items-center gap-3 px-3 py-2.5">
                <div className="min-w-0 flex-1">
                  <div className="font-mono text-[14px]">{t.id}</div>
                  <div className="mt-0.5 flex flex-wrap gap-1">
                    <Badge>{t.family}</Badge>
                    <Badge>{t.tier}</Badge>
                    {t.solved > 0 ? <Badge tone="ok">{triesText(t)}</Badge> : t.attempts > 0 ? <Badge tone="danger">{triesText(t)}</Badge> : null}
                  </div>
                </div>
                <Button disabled={busy} onClick={() => onStart(t.id)}>
                  <Play size={15} /> Start
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function SessionList({ data }: { data: ReOverview }) {
  const [filter, setFilter] = useState<ReOutcome | ''>('')
  const list = data.sessions.filter((s) => !filter || s.outcome === filter)
  if (!data.sessions.length) return <div className="text-[14px] text-muted">No reverse-engineering sessions yet.</div>
  return (
    <div className="space-y-3">
      <Select value={filter} onChange={(e) => setFilter(e.target.value as ReOutcome | '')}>
        <option value="">Every grade</option>
        <option value="solved">Solved</option>
        <option value="partial">Partly solved</option>
        <option value="failed">Failed</option>
        <option value="unrated">Not graded</option>
      </Select>
      <div className="divide-y divide-border rounded-xl border border-border">
        {list.map((s) => (
          <div key={s.id} className="px-3 py-2.5">
            <div className="flex flex-wrap items-center gap-1.5">
              <Badge tone={OUTCOME_TONE[s.outcome]}>{s.outcome === 'unrated' ? 'not graded' : s.outcome}</Badge>
              <Badge>{KIND_LABEL[s.kind]}</Badge>
              <span className="ml-auto text-[12px] text-faint">{relativeTime(s.updatedAt)}</span>
            </div>
            <div className="mt-1 text-[14px]">{s.title}</div>
            <div className="truncate font-mono text-[12px] text-muted">{s.target}</div>
            <div className="text-[12px] text-faint">
              {s.source} · {s.toolRounds} tool rounds
              {s.practice?.expected !== undefined && s.outcome !== 'solved' ? ` · answered ${s.practice.answer ?? '—'}, expected ${s.practice.expected}` : ''}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
