import { Undo2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { MAX_INSTRUCTION_CHARS as MAX_CHARS } from '@shared/defaults'
import type { TaskRecord } from '@shared/types'
import { cn } from '../lib/format'
import { useUi } from '../stores/ui'
import { tasksUndoneBy, useWorkspace } from '../stores/workspace'
import { Badge, Button, Sheet, Textarea } from './ui'

const when = (ts: number): string => {
  const d = new Date(ts)
  const time = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
  return d.toDateString() === new Date().toDateString() ? `Today, ${time}` : `${d.toLocaleDateString([], { day: 'numeric', month: 'short' })}, ${time}`
}

/** Every task in the project, newest first, with "Go back to before this" on each. */
export function HistorySheet({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const tasks = useWorkspace((s) => s.tasks)
  const running = useWorkspace((s) => s.running)
  const undo = useWorkspace((s) => s.undo)
  const selectTask = useWorkspace((s) => s.selectTask)
  const setTab = useUi((s) => s.setTab)
  const [confirm, setConfirm] = useState<TaskRecord | null>(null)
  useEffect(() => {
    if (!open) setConfirm(null)
  }, [open])

  if (confirm) {
    const undone = tasksUndoneBy(tasks, confirm)
    return (
      <Sheet
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
              <Undo2 size={15} /> Go back
            </Button>
          </>
        }
      >
        <div className="px-4 pb-4 text-[15px] text-muted">
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
      </Sheet>
    )
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange} full title="History" description="Every task in this project, newest first. You can put the project back to how it was before any of them.">
      <div className="px-2 pb-4">
        {tasks.length === 0 && <p className="px-3 py-4 text-sm text-faint">No tasks yet. Each task you give me will show up here.</p>}
        {[...tasks].reverse().map((t) => {
          const canGoBack = !running && !t.undone && !!t.snapshot && t.outcome !== 'running'
          return (
            <div key={t.id} className="border-b border-border px-2 py-3 last:border-0">
              <button
                className="w-full text-left"
                onClick={() => {
                  selectTask(t.id === tasks[tasks.length - 1].id ? null : t.id)
                  setTab('steps')
                  onOpenChange(false)
                }}
              >
                <span className={cn('block text-[15px]', t.undone && 'text-muted line-through')}>{t.title}</span>
                <span className="mt-0.5 flex flex-wrap items-center gap-1.5 text-[12px] text-faint">
                  {when(t.startedAt)}
                  {t.files.length > 0 && <span>· {t.files.length === 1 ? '1 file changed' : `${t.files.length} files changed`}</span>}
                  {t.undone && <Badge>Undone</Badge>}
                  {!t.undone && t.outcome === 'running' && <Badge tone="accent">Working now</Badge>}
                  {!t.undone && t.outcome === 'stopped' && <Badge tone="warn">Stopped</Badge>}
                  {!t.undone && t.outcome === 'error' && <Badge tone="danger">Didn’t finish</Badge>}
                </span>
              </button>
              {!t.undone && (
                <Button size="sm" className="mt-2" disabled={!canGoBack} onClick={() => setConfirm(t)}>
                  <Undo2 size={14} /> Go back to before this
                </Button>
              )}
            </div>
          )
        })}
      </div>
    </Sheet>
  )
}

/** "Things to always remember": the owner's notes, sent to the AI with every request. */
export function NotesSheet({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const memory = useWorkspace((s) => s.memory)
  const setMemory = useWorkspace((s) => s.setMemory)
  const [text, setText] = useState(memory)
  useEffect(() => {
    if (open) setText(memory)
  }, [open, memory])

  return (
    <Sheet
      open={open}
      onOpenChange={onOpenChange}
      title="Things to always remember"
      description="Anything I should keep in mind for this project, one per line. I read these before every step."
      footer={
        <>
          <Button onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button
            variant="primary"
            onClick={async () => {
              if (await setMemory(text)) onOpenChange(false)
            }}
          >
            Save
          </Button>
        </>
      }
    >
      <div className="px-4 pb-3">
        <Textarea
          rows={8}
          maxLength={MAX_CHARS}
          aria-label="Things to always remember"
          placeholder={'For example:\nUse blue, not purple.\nKeep everything in one window.'}
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <div className="mt-1 text-right text-[12px] text-faint">
          {text.length.toLocaleString('en-US')} / {MAX_CHARS.toLocaleString('en-US')}
        </div>
      </div>
    </Sheet>
  )
}
