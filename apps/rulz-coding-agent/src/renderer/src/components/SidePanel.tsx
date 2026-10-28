import { Ban, Check, ChevronRight, Circle, FileMinus, FilePen, FilePlus, Hourglass, LoaderCircle, Minus, X } from 'lucide-react'
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { diffLines, withContext } from '@shared/diff'
import type { ChangedFile, FileDiff, Step } from '@shared/types'
import { errorMessage, invoke } from '../lib/api'
import { cn } from '../lib/format'
import { shownTask, useWorkspace } from '../stores/workspace'
import { Badge, Dialog, Select } from './ui'

export function SidePanel() {
  const tasks = useWorkspace((s) => s.tasks)
  const selected = useWorkspace((s) => s.selectedTaskId)
  const select = useWorkspace((s) => s.selectTask)
  const task = shownTask(tasks, selected)
  const [diffFile, setDiffFile] = useState<ChangedFile | null>(null)

  return (
    <aside className="flex w-[380px] shrink-0 flex-col border-l border-border bg-panel">
      <div className="border-b border-border px-4 py-3">
        <div className="text-[11px] font-medium tracking-wide text-faint uppercase">Task</div>
        {tasks.length > 1 ? (
          <Select className="mt-1" value={task?.id ?? ''} onChange={(e) => select(e.target.value === tasks[tasks.length - 1].id ? null : e.target.value)}>
            {[...tasks].reverse().map((t) => (
              <option key={t.id} value={t.id}>
                {t.title}
                {t.undone ? ' (undone)' : ''}
              </option>
            ))}
          </Select>
        ) : (
          <div className="mt-0.5 truncate text-[13px]">{task?.title ?? 'No tasks yet'}</div>
        )}
      </div>

      <section className="flex min-h-0 flex-1 flex-col">
        <h3 className="px-4 pt-3 pb-1.5 text-[13px] font-semibold">Steps</h3>
        <StepList steps={task?.steps ?? []} />
      </section>

      <section className="flex max-h-[40%] min-h-[120px] flex-col border-t border-border">
        <h3 className="flex items-center gap-2 px-4 pt-3 pb-1.5 text-[13px] font-semibold">
          Files changed {task?.files.length ? <Badge>{task.files.length}</Badge> : null}
        </h3>
        <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
          {!task?.files.length ? (
            <p className="px-2 text-xs text-faint">Files I create or change will show up here. Click one to see what changed.</p>
          ) : (
            task.files.map((f) => (
              <button
                key={f.path}
                onClick={() => setDiffFile(f)}
                className="flex w-full items-center gap-2 rounded-md px-2 py-1 text-left text-[13px] hover:bg-hover"
                title={f.path}
              >
                {f.change === 'created' ? (
                  <FilePlus size={14} className="shrink-0 text-ok" />
                ) : f.change === 'deleted' ? (
                  <FileMinus size={14} className="shrink-0 text-danger" />
                ) : (
                  <FilePen size={14} className="shrink-0 text-warn" />
                )}
                <span className={cn('min-w-0 flex-1 truncate', f.change === 'deleted' && 'line-through text-muted')}>{f.path}</span>
                <span className="shrink-0 text-[11px] text-faint">{f.change === 'created' ? 'new' : f.change}</span>
              </button>
            ))
          )}
        </div>
      </section>
      {task && <DiffDialog taskId={task.id} file={diffFile} onClose={() => setDiffFile(null)} />}
    </aside>
  )
}

function StepIcon({ status }: { status: Step['status'] }) {
  switch (status) {
    case 'running':
      return <LoaderCircle size={15} className="animate-spin text-accent" />
    case 'done':
      return <Check size={15} className="text-ok" />
    case 'error':
      return <X size={15} className="text-danger" />
    case 'waiting':
      return <Hourglass size={14} className="text-warn" />
    case 'refused':
      return <Ban size={14} className="text-danger" />
    case 'skipped':
      return <Minus size={15} className="text-faint" />
    default:
      return <Circle size={12} className="text-faint" />
  }
}

function StepList({ steps }: { steps: Step[] }) {
  const ref = useRef<HTMLDivElement>(null)
  const stick = useRef(true)
  useLayoutEffect(() => {
    if (ref.current && stick.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [steps])

  return (
    <div
      ref={ref}
      className="min-h-0 flex-1 overflow-y-auto px-2 pb-2"
      onScroll={(e) => {
        const el = e.currentTarget
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
      }}
    >
      {steps.length === 0 ? (
        <p className="px-2 text-xs text-faint">What I'm doing will show up here, step by step.</p>
      ) : (
        steps.map((s) => <StepRow key={s.id} step={s} />)
      )}
    </div>
  )
}

const STATUS_WORD: Partial<Record<Step['status'], string>> = {
  waiting: 'Waiting for your OK',
  error: 'Didn’t work; I’ll look at why',
  refused: 'Not allowed',
  skipped: 'Skipped'
}

function StepRow({ step }: { step: Step }) {
  const [open, setOpen] = useState(false)
  const outRef = useRef<HTMLPreElement>(null)
  useEffect(() => {
    if (open && outRef.current) outRef.current.scrollTop = outRef.current.scrollHeight
  }, [open, step.output])
  const note = step.note ?? STATUS_WORD[step.status]

  return (
    <div className={cn('rounded-md', open && 'bg-panel-2')}>
      <button className="flex w-full items-start gap-2 rounded-md px-2 py-1.5 text-left hover:bg-hover" onClick={() => setOpen(!open)}>
        <span className="mt-0.5 flex size-4 shrink-0 items-center justify-center">
          <StepIcon status={step.status} />
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-[13px] leading-snug">{step.title}</span>
          {note && step.status !== 'done' && step.status !== 'running' && <span className="block text-[11px] text-faint">{note}</span>}
        </span>
        <ChevronRight size={13} className={cn('mt-1 shrink-0 text-faint transition-transform', open && 'rotate-90')} aria-label="Show details" />
      </button>
      {open && (
        <div className="space-y-1.5 px-2 pb-2">
          {step.detail && <pre className="overflow-x-auto rounded border border-border bg-code px-2 py-1 font-mono text-[11.5px] whitespace-pre-wrap">{step.detail}</pre>}
          {step.output ? (
            <pre ref={outRef} className="max-h-56 overflow-auto rounded border border-border bg-code px-2 py-1 font-mono text-[11.5px] whitespace-pre-wrap text-muted">
              {step.output}
            </pre>
          ) : (
            step.status === 'running' && <p className="text-[11px] text-faint">No output yet.</p>
          )}
        </div>
      )}
    </div>
  )
}

function DiffDialog({ taskId, file, onClose }: { taskId: string; file: ChangedFile | null; onClose: () => void }) {
  const project = useWorkspace((s) => s.project)
  const [diff, setDiff] = useState<FileDiff | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    setDiff(null)
    setError('')
    if (!file || !project) return
    invoke('files:diff', project.path, taskId, file.path).then(setDiff, (e) => setError(errorMessage(e)))
  }, [file, project, taskId])

  const lines = useMemo(() => {
    if (!diff || diff.binary || diff.tooLarge) return null
    const all = diffLines(diff.before ?? '', diff.after ?? '')
    return file?.change === 'changed' ? withContext(all, 4) : all
  }, [diff, file])

  return (
    <Dialog
      open={!!file}
      onOpenChange={(o) => !o && onClose()}
      title={file?.path ?? ''}
      description={file?.change === 'created' ? 'New file' : file?.change === 'deleted' ? 'Deleted file' : 'Changed file: green lines were added, red lines removed'}
      className="w-[min(960px,calc(100vw-48px))]"
    >
      <div className="p-4">
        {error && <p className="text-sm text-danger">{error}</p>}
        {!diff && !error && <p className="text-sm text-faint">Loading…</p>}
        {diff?.binary && <p className="text-sm text-muted">This isn't a text file, so there's nothing to show here.</p>}
        {diff?.tooLarge && <p className="text-sm text-muted">This file is too large to show here.</p>}
        {lines && (
          <div className="overflow-x-auto rounded-lg border border-border bg-code font-mono text-[12px] leading-relaxed">
            {lines.length === 0 && <p className="p-3 text-faint">(empty file)</p>}
            {lines.map((l, i) =>
              l === null ? (
                <div key={i} className="border-y border-border bg-panel-2 px-3 py-0.5 text-[11px] text-faint">
                  ⋯
                </div>
              ) : (
                <div key={i} className={cn('flex', l.kind === 'add' && 'diff-add', l.kind === 'del' && 'diff-del')}>
                  <span className="w-10 shrink-0 pr-2 text-right text-faint select-none">{l.kind === 'del' ? l.oldNo : l.newNo}</span>
                  <span className="w-4 shrink-0 text-faint select-none">{l.kind === 'add' ? '+' : l.kind === 'del' ? '−' : ''}</span>
                  <span className="pr-3 whitespace-pre">{l.text}</span>
                </div>
              )
            )}
          </div>
        )}
      </div>
    </Dialog>
  )
}
