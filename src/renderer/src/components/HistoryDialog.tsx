import { Undo2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { TaskRecord } from '@shared/types'
import { cn } from '../lib/format'
import { tasksUndoneBy, useWorkspace } from '../stores/workspace'
import { Badge, Button, Dialog } from './ui'

const when = (ts: number): string => {
  const d = new Date(ts)
  const time = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
  return d.toDateString() === new Date().toDateString() ? `Today, ${time}` : `${d.toLocaleDateString([], { day: 'numeric', month: 'short' })}, ${time}`
}

/** Every task in the project, newest first, with "Go back to before this" on each. */
export function HistoryDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const tasks = useWorkspace((s) => s.tasks)
  const running = useWorkspace((s) => s.running)
  const undo = useWorkspace((s) => s.undo)
  const select = useWorkspace((s) => s.selectTask)
  const [confirm, setConfirm] = useState<TaskRecord | null>(null)
  useEffect(() => {
    if (!open) setConfirm(null)
  }, [open])

  if (confirm) {
    const undone = tasksUndoneBy(tasks, confirm)
    return (
      <Dialog
        open={open}
        onOpenChange={onOpenChange}
        title="Go back to before this task?"
        footer={
          <>
            <Button onClick={() => setConfirm(null)}>Cancel</Button>
            <Button
              variant="primary"
              onClick={() => {
                onOpenChange(false)
                void undo(confirm.id)
              }}
            >
              <Undo2 size={14} /> Go back
            </Button>
          </>
        }
      >
        <div className="px-5 py-4 text-sm text-muted">
          <p>
            This puts your project back to exactly how it was before <b className="text-text">“{confirm.title}”</b>.
            {undone.length > 1 ? ` Everything these ${undone.length} tasks added or changed will be removed:` : ' Everything that task added or changed will be removed.'}
          </p>
          {undone.length > 1 && (
            <ul className="mt-2 list-disc space-y-0.5 pl-5">
              {undone
                .slice()
                .reverse()
                .map((t) => (
                  <li key={t.id} className="text-text">
                    {t.title}
                  </li>
                ))}
            </ul>
          )}
        </div>
      </Dialog>
    )
  }

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="History"
      description="Every task in this project, newest first. You can put the project back to how it was before any of them."
      className="w-[min(640px,calc(100vw-48px))]"
    >
      <div className="max-h-[60vh] overflow-y-auto p-2">
        {tasks.length === 0 && <p className="px-3 py-4 text-sm text-faint">No tasks yet. Each task you give me will show up here.</p>}
        {[...tasks].reverse().map((t) => {
          const canGoBack = !running && !t.undone && !!t.snapshot && t.outcome !== 'running'
          return (
            <div key={t.id} className="flex items-center gap-3 rounded-lg px-3 py-2.5 hover:bg-hover" data-testid="history-row">
              <button
                className="min-w-0 flex-1 text-left"
                title="Show this task's steps and files"
                onClick={() => {
                  select(t.id === tasks[tasks.length - 1].id ? null : t.id)
                  onOpenChange(false)
                }}
              >
                <span className={cn('block truncate text-[14px]', t.undone && 'text-muted line-through')}>{t.title}</span>
                <span className="mt-0.5 flex flex-wrap items-center gap-1.5 text-[11px] text-faint">
                  {when(t.startedAt)}
                  {t.files.length > 0 && <span>· {t.files.length === 1 ? '1 file changed' : `${t.files.length} files changed`}</span>}
                  {t.undone && <Badge>Undone</Badge>}
                  {!t.undone && t.outcome === 'running' && <Badge tone="accent">Working now</Badge>}
                  {!t.undone && t.outcome === 'stopped' && <Badge tone="warn">Stopped</Badge>}
                  {!t.undone && t.outcome === 'error' && <Badge tone="danger">Didn’t finish</Badge>}
                </span>
              </button>
              {!t.undone && (
                <Button
                  size="sm"
                  disabled={!canGoBack}
                  title={t.snapshot ? undefined : 'No restore point was saved for this task'}
                  onClick={() => setConfirm(t)}
                >
                  <Undo2 size={13} /> Go back to before this
                </Button>
              )}
            </div>
          )
        })}
      </div>
    </Dialog>
  )
}
