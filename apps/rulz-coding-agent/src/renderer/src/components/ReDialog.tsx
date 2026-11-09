import { LoaderCircle, Play, RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'
import type { ReKind, ReOutcome, ReOverview, ReStartRequest } from '@shared/types'
import { errorMessage, invoke } from '../lib/api'
import { cn, relativeTime } from '../lib/format'
import { useWorkspace } from '../stores/workspace'
import { Badge, Button, Dialog, Field, Input, Select, Switch, Textarea } from './ui'

export const OUTCOME_TONE: Record<ReOutcome, 'neutral' | 'ok' | 'warn' | 'danger'> = { unrated: 'neutral', solved: 'ok', partial: 'warn', failed: 'danger' }
export const KIND_LABEL: Record<ReKind, string> = { analyse: 'Analyse', practice: 'Practice', 'clone-pieces': 'Clone pieces', 'clone-full': 'Full clone' }

/** RulZ LLM Suite's RE overview, loaded when asked for. */
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
      .catch((e) => setError(errorMessage(e)))
      .finally(() => setLoading(false))
  }, [])
  useEffect(() => {
    if (open) reload()
  }, [open, reload])
  return { data, error, loading, reload }
}

type Tab = 'start' | 'practice' | 'sessions'

/** Reverse engineering: start the tasks RulZ LLM Suite's RE page offers, and see what was done. */
export function ReDialog({ open, onOpenChange, initialTab = 'start' }: { open: boolean; onOpenChange: (open: boolean) => void; initialTab?: Tab }) {
  const project = useWorkspace((s) => s.project)
  const running = useWorkspace((s) => s.running)
  const { data, error, loading, reload } = useReOverview(open)
  const [tab, setTab] = useState<Tab>(initialTab)
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
      toast.error(errorMessage(e))
    }
  }

  const solved = data?.sessions.filter((s) => s.outcome === 'solved').length ?? 0
  return (
    <Dialog open={open} onOpenChange={onOpenChange} title="Reverse engineering" className="w-[min(760px,calc(100vw-32px))]">
      <div className="flex items-center gap-1 border-b border-border px-5 pb-2">
        {(
          [
            ['start', 'Start a task'],
            ['practice', `Practice${data ? ` (${data.practice.tasks.length})` : ''}`],
            ['sessions', `Done${data ? ` (${data.sessions.length})` : ''}`]
          ] as const
        ).map(([value, label]) => (
          <button
            key={value}
            onClick={() => setTab(value)}
            className={cn('rounded-md px-3 py-1 text-[13px]', tab === value ? 'bg-accent-soft text-accent' : 'text-muted hover:bg-hover hover:text-text')}
          >
            {label}
          </button>
        ))}
        <div className="flex-1" />
        {data && (
          <span className="text-[11px] text-faint">
            {solved} solved · recording {data.recording ? 'on' : 'off'} in RulZ LLM Suite
          </span>
        )}
        <button className="rounded p-1 text-muted hover:bg-hover hover:text-text" aria-label="Refresh" onClick={reload}>
          {loading ? <LoaderCircle size={14} className="animate-spin" /> : <RefreshCw size={14} />}
        </button>
      </div>
      <div className="min-h-[320px] overflow-y-auto px-5 py-4">
        {error ? (
          <div className="rounded-lg border border-warn/40 bg-warn/10 p-3 text-sm">{error}</div>
        ) : !data ? (
          <div className="flex items-center gap-2 text-sm text-muted">
            <LoaderCircle size={14} className="animate-spin" /> Asking RulZ LLM Suite…
          </div>
        ) : tab === 'start' ? (
          <StartForm data={data} busy={running} onStart={start} />
        ) : tab === 'practice' ? (
          <PracticeList data={data} busy={running} onStart={(taskId) => void start({ kind: 'practice', taskId })} />
        ) : (
          <SessionList data={data} />
        )}
      </div>
      <div className="flex items-center gap-2 border-t border-border px-5 py-3 text-[12px] text-muted">
        <Switch checked={record} onCheckedChange={setRecord} />
        Record for training in RulZ LLM Suite (graded on its Reverse Engineering page)
        {running && <span className="ml-auto text-warn">A task is running; wait for it to finish.</span>}
      </div>
    </Dialog>
  )
}

function Chips({ items, value, onPick }: { items: { label: string; text: string }[]; value: string; onPick: (t: string) => void }) {
  return (
    <div className="mb-1.5 flex flex-wrap gap-1">
      {items.map((g) => (
        <button
          key={g.label}
          onClick={() => onPick(g.text)}
          className={cn(
            'rounded-md border px-2 py-0.5 text-[11.5px]',
            value === g.text ? 'border-accent/50 bg-accent-soft text-accent' : 'border-border text-muted hover:bg-hover hover:text-text'
          )}
        >
          {g.label}
        </button>
      ))}
    </div>
  )
}

function StartForm({ data, busy, onStart }: { data: ReOverview; busy: boolean; onStart: (req: ReStartRequest) => Promise<void> }) {
  const [mode, setMode] = useState<'analyse' | 'clone'>('analyse')
  const [target, setTarget] = useState('')
  const [goal, setGoal] = useState(data.goals[0]?.text ?? '')
  const [scope, setScope] = useState<'pieces' | 'full'>('pieces')
  const [pieces, setPieces] = useState(data.clonePieces[0]?.text ?? '')
  const [notes, setNotes] = useState('')
  const [language, setLanguage] = useState('')
  const [starting, setStarting] = useState(false)
  const pick = async (kind: 'file' | 'folder'): Promise<void> => {
    const title = 'Choose the program or app folder'
    const p = await invoke(kind === 'file' ? 'dialog:pickFile' : 'dialog:pickFolder', { title })
    if (p) setTarget(p)
  }
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
      <div className="flex gap-1 rounded-lg border border-border p-0.5 text-[13px]">
        {(
          [
            ['analyse', 'Analyse: explain it, find answers'],
            ['clone', 'Clone: rebuild it as new code here']
          ] as const
        ).map(([v, l]) => (
          <button key={v} onClick={() => setMode(v)} className={cn('flex-1 rounded-md px-3 py-1', mode === v ? 'bg-accent-soft text-accent' : 'text-muted hover:bg-hover')}>
            {l}
          </button>
        ))}
      </div>
      <Field label="Target">
        <div className="flex gap-2">
          <Input className="font-mono text-xs" value={target} placeholder="C:\path\to\program.exe" onChange={(e) => setTarget(e.target.value)} />
          <Button onClick={() => void pick('file')}>File…</Button>
          <Button onClick={() => void pick('folder')}>Folder…</Button>
        </div>
      </Field>
      {mode === 'analyse' ? (
        <Field label="What to find out" hint="{target} is replaced by the path.">
          <Chips items={data.goals} value={goal} onPick={setGoal} />
          <Textarea rows={3} value={goal} onChange={(e) => setGoal(e.target.value)} />
        </Field>
      ) : (
        <>
          <Field label="What to clone">
            <div className="mb-1.5 flex gap-4 text-[13px]">
              {(
                [
                  ['pieces', 'Specific pieces'],
                  ['full', 'The whole program']
                ] as const
              ).map(([v, l]) => (
                <label key={v} className="flex items-center gap-1.5">
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
          <Field label="Language" hint="The new code goes in this project's folder.">
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
      <div className="flex items-center justify-end gap-3">
        <span className="text-[11px] text-faint">Uses the REA tools set up in RulZ LLM Suite. Only work on software you have the right to.</span>
        <Button variant="primary" disabled={!ready || busy || starting} onClick={() => void go()}>
          <Play size={14} /> {mode === 'clone' ? 'Start cloning' : 'Start'}
        </Button>
      </div>
    </div>
  )
}

function PracticeList({ data, busy, onStart }: { data: ReOverview; busy: boolean; onStart: (taskId: string) => void }) {
  const [family, setFamily] = useState('')
  const [unsolved, setUnsolved] = useState(false)
  const families = useMemo(() => [...new Set(data.practice.tasks.map((t) => t.family))].sort(), [data])
  const tasks = data.practice.tasks.filter((t) => (!family || t.family === family) && (!unsolved || t.solved === 0))
  if (data.practice.error) return <div className="rounded-lg border border-warn/40 bg-warn/10 p-3 text-sm">{data.practice.error}</div>
  if (!data.practice.tasks.length) return <div className="text-sm text-muted">No practice tasks in {data.practice.kitDir}.</div>
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3 text-[13px]">
        <Select value={family} onChange={(e) => setFamily(e.target.value)} className="w-48">
          <option value="">Every family</option>
          {families.map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </Select>
        <label className="flex items-center gap-2">
          <Switch checked={unsolved} onCheckedChange={setUnsolved} /> Not solved yet
        </label>
        <span className="ml-auto text-[11px] text-faint">
          {tasks.length} of {data.practice.tasks.length} tasks
        </span>
      </div>
      <div className="divide-y divide-border rounded-lg border border-border">
        {tasks.map((t) => (
          <div key={t.id} className="flex items-center gap-3 px-3 py-2">
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 text-[13px]">
                <span className="font-mono">{t.id}</span>
                <Badge>{t.family}</Badge>
                <Badge>{t.tier}</Badge>
                {t.solved > 0 ? <Badge tone="ok">solved {t.solved}/{t.attempts}</Badge> : t.attempts > 0 ? <Badge tone="danger">0/{t.attempts}</Badge> : null}
              </div>
              <div className="truncate text-[11.5px] text-muted" title={t.prompt}>
                {t.prompt.replaceAll('{exe}', t.binary)}
              </div>
            </div>
            <Button disabled={busy} onClick={() => onStart(t.id)}>
              <Play size={13} /> Start
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
  if (!data.sessions.length) return <div className="text-sm text-muted">No reverse-engineering sessions yet.</div>
  return (
    <div className="space-y-3">
      <Select value={filter} onChange={(e) => setFilter(e.target.value as ReOutcome | '')} className="w-44">
        <option value="">Every grade</option>
        <option value="solved">Solved</option>
        <option value="partial">Partly solved</option>
        <option value="failed">Failed</option>
        <option value="unrated">Not graded</option>
      </Select>
      <div className="divide-y divide-border rounded-lg border border-border">
        {list.map((s) => (
          <div key={s.id} className="px-3 py-2">
            <div className="flex items-center gap-2 text-[13px]">
              <Badge tone={OUTCOME_TONE[s.outcome]}>{s.outcome === 'unrated' ? 'not graded' : s.outcome}</Badge>
              <Badge>{KIND_LABEL[s.kind]}</Badge>
              <span className="min-w-0 flex-1 truncate">{s.title}</span>
              <span className="shrink-0 text-[11px] text-faint">{relativeTime(s.updatedAt)}</span>
            </div>
            <div className="mt-0.5 flex gap-3 text-[11.5px] text-muted">
              <span className="min-w-0 flex-1 truncate font-mono" title={s.target}>
                {s.target}
              </span>
              <span className="shrink-0">{s.source}</span>
              <span className="shrink-0">{s.toolRounds} tool rounds</span>
            </div>
            {s.practice?.expected !== undefined && s.outcome !== 'solved' && (
              <div className="text-[11.5px] text-faint">
                answered {s.practice.answer ?? '—'} · expected {s.practice.expected}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
