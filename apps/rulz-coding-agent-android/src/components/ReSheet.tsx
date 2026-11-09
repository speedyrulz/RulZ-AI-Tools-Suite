import { Folder, Play, RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import type { ReKind, ReOutcome, ReOverview, ReStartRequest } from '@shared/types'
import { toast } from 'sonner'
import { errorMessage, invoke } from '../lib/api'
import { cn, relativeTime } from '../lib/format'
import { useWorkspace } from '../stores/workspace'
import { FolderPicker } from './FolderPicker'
import { Badge, Button, Field, Input, Select, Sheet, Spinner, Switch, Textarea } from './ui'

const OUTCOME_TONE: Record<ReOutcome, 'neutral' | 'ok' | 'warn' | 'danger'> = { unrated: 'neutral', solved: 'ok', partial: 'warn', failed: 'danger' }
const KIND_LABEL: Record<ReKind, string> = { analyse: 'Analyse', practice: 'Practice', 'clone-pieces': 'Clone pieces', 'clone-full': 'Full clone' }

/** A computer running an older RulZ Coding Agent doesn't know the RE requests. */
function friendly(e: unknown): string {
  const m = errorMessage(e)
  return /re:(overview|start)/.test(m) ? 'Update RulZ Coding Agent on the computer to 0.18.0 or later to use reverse engineering from the phone.' : m
}

/** RulZ LLM Suite's RE overview, through the computer. */
export function useReOverview(open: boolean): { data: ReOverview | null; error: string | null; loading: boolean; reload: () => void } {
  const [data, setData] = useState<ReOverview | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const reload = useCallback(() => {
    setLoading(true)
    invoke('re:overview')
      .then((d) => {
        setData(d)
        setError(null)
      })
      .catch((e) => setError(friendly(e)))
      .finally(() => setLoading(false))
  }, [])
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
  const { data, error, loading, reload } = useReOverview(open)
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
        {running && <div className="mb-3 text-[13px] text-warn">A task is running; wait for it to finish before starting another.</div>}
        {error ? (
          <div className="rounded-xl border border-warn/40 bg-warn/10 p-3 text-[14px]">{error}</div>
        ) : !data ? (
          <div className="flex items-center gap-2 text-[14px] text-muted">
            <Spinner /> Asking RulZ LLM Suite…
          </div>
        ) : tab === 'start' ? (
          <StartForm data={data} busy={running} onStart={start} />
        ) : tab === 'practice' ? (
          <PracticeList data={data} busy={running} onStart={(taskId) => void start({ kind: 'practice', taskId })} />
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

function PracticeList({ data, busy, onStart }: { data: ReOverview; busy: boolean; onStart: (taskId: string) => void }) {
  const [family, setFamily] = useState('')
  const [unsolved, setUnsolved] = useState(false)
  const families = useMemo(() => [...new Set(data.practice.tasks.map((t) => t.family))].sort(), [data])
  const tasks = data.practice.tasks.filter((t) => (!family || t.family === family) && (!unsolved || t.solved === 0))
  if (data.practice.error) return <div className="rounded-xl border border-warn/40 bg-warn/10 p-3 text-[14px]">{data.practice.error}</div>
  if (!data.practice.tasks.length) return <div className="text-[14px] text-muted">No practice tasks in {data.practice.kitDir}.</div>
  return (
    <div className="space-y-3">
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
        <span className="ml-auto text-[12px] text-faint">{tasks.length} tasks</span>
      </label>
      <div className="divide-y divide-border rounded-xl border border-border">
        {tasks.map((t) => (
          <div key={t.id} className="flex items-center gap-3 px-3 py-2.5">
            <div className="min-w-0 flex-1">
              <div className="font-mono text-[14px]">{t.id}</div>
              <div className="mt-0.5 flex flex-wrap gap-1">
                <Badge>{t.family}</Badge>
                <Badge>{t.tier}</Badge>
                {t.solved > 0 ? <Badge tone="ok">solved {t.solved}/{t.attempts}</Badge> : t.attempts > 0 ? <Badge tone="danger">0/{t.attempts}</Badge> : null}
              </div>
            </div>
            <Button disabled={busy} onClick={() => onStart(t.id)}>
              <Play size={15} /> Start
            </Button>
          </div>
        ))}
      </div>
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
