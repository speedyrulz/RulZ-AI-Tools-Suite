import { Ban, Check, ChevronRight, Circle, Gauge, FileMinus, FilePen, FilePlus, Hourglass, LoaderCircle, Minus, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { diffLines, withContext } from '@shared/diff'
import type { ChangedFile, FileDiff, Step, TaskRecord } from '@shared/types'
import { errorMessage, invoke } from '../lib/api'
import { draftSummary } from '../lib/drafts'
import { cn } from '../lib/format'
import { shownTask, useWorkspace } from '../stores/workspace'
import { Badge, Select, Sheet } from './ui'

/** Which task's steps and files are shown, when there is more than one. */
export function TaskPicker() {
  const tasks = useWorkspace((s) => s.tasks)
  const selected = useWorkspace((s) => s.selectedTaskId)
  const select = useWorkspace((s) => s.selectTask)
  const task = shownTask(tasks, selected)
  if (tasks.length < 2) return task ? <div className="truncate px-4 pt-3 text-[13px] text-muted">{task.title}</div> : null
  return (
    <div className="px-4 pt-3">
      <Select value={task?.id ?? ''} onChange={(e) => select(e.target.value === tasks[tasks.length - 1].id ? null : e.target.value)} aria-label="Task">
        {[...tasks].reverse().map((t) => (
          <option key={t.id} value={t.id}>
            {t.title}
            {t.undone ? ' (undone)' : ''}
          </option>
        ))}
      </Select>
    </div>
  )
}

/** Speculative decoding (MTP / draft model) in the shown task: how many guesses the model kept, and the time that saved. Tap for the counts. */
export function DraftReadout() {
  const tasks = useWorkspace((s) => s.tasks)
  const selected = useWorkspace((s) => s.selectedTaskId)
  const [open, setOpen] = useState(false)
  const summary = draftSummary(shownTask(tasks, selected)?.draft)
  if (!summary) return null
  return (
    <div className="px-4 pt-2 text-[13px] text-faint">
      <button
        type="button"
        className="flex items-center gap-1.5 tabular-nums"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-label={`Drafts ${summary.rate}% accepted, ${summary.saved}`}
      >
        <Gauge size={14} />
        <span>Drafts</span>
        <span>
          {summary.rate}% · {summary.saved}
        </span>
        <ChevronRight size={13} className={cn('transition-transform', open && 'rotate-90')} />
      </button>
      {open && (
        <div className="mt-1.5 space-y-1 rounded-lg bg-panel-2 px-3 py-2 text-muted">
          <p>Speculative decoding (MTP) in this task: {summary.detail}</p>
          <p className="text-[12px] text-faint">{summary.how}</p>
        </div>
      )}
    </div>
  )
}

function StepIcon({ status }: { status: Step['status'] }) {
  switch (status) {
    case 'running':
      return <LoaderCircle size={17} className="animate-spin text-accent" />
    case 'done':
      return <Check size={17} className="text-ok" />
    case 'error':
      return <X size={17} className="text-danger" />
    case 'waiting':
      return <Hourglass size={16} className="text-warn" />
    case 'refused':
      return <Ban size={16} className="text-danger" />
    case 'skipped':
      return <Minus size={17} className="text-faint" />
    default:
      return <Circle size={13} className="text-faint" />
  }
}

const STATUS_WORD: Partial<Record<Step['status'], string>> = {
  waiting: 'Waiting for your OK',
  error: 'Didn’t work; I’ll look at why',
  refused: 'Not allowed',
  skipped: 'Skipped'
}

export function StepList() {
  const tasks = useWorkspace((s) => s.tasks)
  const selected = useWorkspace((s) => s.selectedTaskId)
  const steps = shownTask(tasks, selected)?.steps ?? []
  if (!steps.length) return <p className="px-4 py-6 text-sm text-faint">What I'm doing will show up here, step by step.</p>
  return (
    <div className="px-2 py-2">
      {steps.map((s) => (
        <StepRow key={s.id} step={s} />
      ))}
    </div>
  )
}

function StepRow({ step }: { step: Step }) {
  const [open, setOpen] = useState(false)
  const outRef = useRef<HTMLPreElement>(null)
  useEffect(() => {
    if (open && outRef.current) outRef.current.scrollTop = outRef.current.scrollHeight
  }, [open, step.output])
  const note = step.note ?? STATUS_WORD[step.status]

  return (
    <div className={cn('rounded-lg', open && 'bg-panel-2')}>
      <button className="flex w-full items-start gap-2.5 rounded-lg px-2 py-2.5 text-left active:bg-hover" onClick={() => setOpen(!open)}>
        <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center">
          <StepIcon status={step.status} />
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-[15px] leading-snug">{step.title}</span>
          {note && step.status !== 'done' && step.status !== 'running' && <span className="block text-[12px] text-faint">{note}</span>}
        </span>
        <ChevronRight size={15} className={cn('mt-1 shrink-0 text-faint transition-transform', open && 'rotate-90')} aria-label="Show details" />
      </button>
      {open && (
        <div className="space-y-1.5 px-2 pb-2">
          {step.detail && <pre className="overflow-x-auto rounded border border-border bg-code px-2 py-1.5 font-mono text-[12px] whitespace-pre-wrap">{step.detail}</pre>}
          {step.output ? (
            <pre ref={outRef} className="max-h-72 overflow-auto rounded border border-border bg-code px-2 py-1.5 font-mono text-[12px] whitespace-pre-wrap text-muted">
              {step.output}
            </pre>
          ) : (
            step.status === 'running' && <p className="text-[12px] text-faint">No output yet.</p>
          )}
        </div>
      )}
    </div>
  )
}

export function FileList() {
  const tasks = useWorkspace((s) => s.tasks)
  const selected = useWorkspace((s) => s.selectedTaskId)
  const task = shownTask(tasks, selected)
  const [diffFile, setDiffFile] = useState<ChangedFile | null>(null)

  if (!task?.files.length) return <p className="px-4 py-6 text-sm text-faint">Files I create or change will show up here. Tap one to see what changed.</p>
  return (
    <div className="px-2 py-2">
      {task.files.map((f) => (
        <button key={f.path} onClick={() => setDiffFile(f)} className="flex w-full items-center gap-2.5 rounded-lg px-2 py-3 text-left text-[14px] active:bg-hover">
          {f.change === 'created' ? (
            <FilePlus size={17} className="shrink-0 text-ok" />
          ) : f.change === 'deleted' ? (
            <FileMinus size={17} className="shrink-0 text-danger" />
          ) : (
            <FilePen size={17} className="shrink-0 text-warn" />
          )}
          <span className={cn('min-w-0 flex-1 break-all', f.change === 'deleted' && 'text-muted line-through')}>{f.path}</span>
          <span className="shrink-0 text-[12px] text-faint">{f.change === 'created' ? 'new' : f.change}</span>
        </button>
      ))}
      <DiffSheet task={task} file={diffFile} onClose={() => setDiffFile(null)} />
    </div>
  )
}

function DiffSheet({ task, file, onClose }: { task: TaskRecord; file: ChangedFile | null; onClose: () => void }) {
  const project = useWorkspace((s) => s.project)
  const [diff, setDiff] = useState<FileDiff | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    setDiff(null)
    setError('')
    if (!file || !project) return
    invoke('files:diff', project.path, task.id, file.path).then(setDiff, (e) => setError(errorMessage(e)))
  }, [file, project, task.id])

  const lines = useMemo(() => {
    if (!diff || diff.binary || diff.tooLarge) return null
    const all = diffLines(diff.before ?? '', diff.after ?? '')
    return file?.change === 'changed' ? withContext(all, 3) : all
  }, [diff, file])

  return (
    <Sheet
      open={!!file}
      onOpenChange={(o) => !o && onClose()}
      full
      title={file?.path ?? ''}
      description={file?.change === 'created' ? 'New file' : file?.change === 'deleted' ? 'Deleted file' : 'Green lines were added, red lines removed'}
    >
      <div className="px-2 pb-4">
        {error && <p className="px-2 text-sm text-danger">{error}</p>}
        {!diff && !error && <p className="px-2 text-sm text-faint">Loading…</p>}
        {diff?.binary && <p className="px-2 text-sm text-muted">This isn't a text file, so there's nothing to show here.</p>}
        {diff?.tooLarge && <p className="px-2 text-sm text-muted">This file is too large to show here.</p>}
        {lines && (
          <div className="overflow-x-auto rounded-lg border border-border bg-code font-mono text-[11.5px] leading-relaxed">
            {lines.length === 0 && <p className="p-3 text-faint">(empty file)</p>}
            {lines.map((l, i) =>
              l === null ? (
                <div key={i} className="border-y border-border bg-panel-2 px-3 py-0.5 text-[11px] text-faint">
                  ⋯
                </div>
              ) : (
                <div key={i} className={cn('flex w-max min-w-full', l.kind === 'add' && 'diff-add', l.kind === 'del' && 'diff-del')}>
                  <span className="w-9 shrink-0 pr-1.5 text-right text-faint select-none">{l.kind === 'del' ? l.oldNo : l.newNo}</span>
                  <span className="w-3.5 shrink-0 text-faint select-none">{l.kind === 'add' ? '+' : l.kind === 'del' ? '−' : ''}</span>
                  <span className="pr-3 whitespace-pre">{l.text}</span>
                </div>
              )
            )}
          </div>
        )}
      </div>
    </Sheet>
  )
}

/** Small count for the Files tab. */
export function FileCount() {
  const tasks = useWorkspace((s) => s.tasks)
  const selected = useWorkspace((s) => s.selectedTaskId)
  const n = shownTask(tasks, selected)?.files.length ?? 0
  return n ? <Badge className="px-1 py-0 text-[10px]">{n}</Badge> : null
}
