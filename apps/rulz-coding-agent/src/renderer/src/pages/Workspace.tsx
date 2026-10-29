import { ArrowLeft, ArrowUp, Brain, Check, ChevronDown, ChevronRight, Clock, Copy, CornerDownRight, FolderOpen, SlidersHorizontal, History, LoaderCircle, MessageSquarePlus, NotebookPen, Package, Pencil, Play, RotateCcw, Settings, Square, Trash2, Undo2, X } from 'lucide-react'
import { toast } from 'sonner'
import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { ApprovalDialog } from '../components/ApprovalDialog'
import { ContextMeter } from '../components/ContextMeter'
import { DraftMeter } from '../components/DraftMeter'
import { HistoryDialog } from '../components/HistoryDialog'
import { NotesDialog } from '../components/NotesDialog'
import { LogoMark } from '../components/Logo'
import { Markdown } from '../components/Markdown'
import { ConnectionHelp } from '../components/NotConnected'
import { RunPanel } from '../components/RunPanel'
import { SidePanel } from '../components/SidePanel'
import { modelLabel } from '../components/common'
import { Button, Dialog, DropdownMenu, IconButton, Kbd, Textarea, Tooltip } from '../components/ui'
import { invoke } from '../lib/api'
import { cn } from '../lib/format'
import { activeModelId, useApp } from '../stores/app'
import { undoableTask, useWorkspace } from '../stores/workspace'
import { ThemeToggle } from './Home'
import { whyNotDelete, whyNotRedo } from '@shared/history'
import type { TimelineItem } from '@shared/types'
import { splitThink, thinkingOpen } from '@shared/think'

const EXAMPLES = [
  'Make a simple to-do list desktop app',
  'Make a calculator website with big buttons',
  'Make a video upscaler with a drag-and-drop window'
]

export function Workspace() {
  const connected = useApp((s) => s.connection.status === 'connected')
  const running = useWorkspace((s) => s.running)
  const prompt = useWorkspace((s) => s.prompt)
  const stop = useWorkspace((s) => s.stop)

  // Esc stops the task (dialogs handle their own Esc).
  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape' && running && prompt?.kind !== 'approval' && !document.querySelector('[role="dialog"]')) stop()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [running, prompt, stop])

  return (
    <div className="flex h-full flex-col">
      <TopBar />
      {!connected && (
        <div className="border-b border-border bg-panel-2 px-6 py-4">
          <ConnectionHelp compact />
        </div>
      )}
      <div className="flex min-h-0 flex-1">
        <div className="flex min-w-0 flex-1 flex-col">
          <Conversation />
          <PromptCard />
          <Composer />
          <RunPanel />
        </div>
        <SidePanel />
      </div>
      <ApprovalDialog />
    </div>
  )
}

function TopBar() {
  const project = useWorkspace((s) => s.project)!
  const running = useWorkspace((s) => s.running)
  const tasks = useWorkspace((s) => s.tasks)
  const run = useWorkspace((s) => s.run)
  const close = useWorkspace((s) => s.close)
  const startRun = useWorkspace((s) => s.startRun)
  const stopRun = useWorkspace((s) => s.stopRun)
  const undo = useWorkspace((s) => s.undo)
  const share = useWorkspace((s) => s.share)
  const memory = useWorkspace((s) => s.memory)
  const setSettingsOpen = useApp((s) => s.setSettingsOpen)
  const [confirmUndo, setConfirmUndo] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [notesOpen, setNotesOpen] = useState(false)
  const [confirmShare, setConfirmShare] = useState(false)
  const undoTask = undoableTask(tasks)
  // Labels show on wide windows; narrow ones keep the icons (the aria-label names the button).
  const label = (text: string) => <span className="hidden xl:inline">{text}</span>

  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b border-border bg-panel px-3">
      <Tooltip content="Back to your projects">
        <button
          className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-muted hover:bg-hover hover:text-text disabled:opacity-40"
          onClick={() => void close()}
          disabled={running}
          aria-label="Back to your projects"
        >
          <ArrowLeft size={16} />
          <LogoMark className="w-7" />
        </button>
      </Tooltip>
      <div className="min-w-0">
        <div className="truncate text-[15px] font-semibold">{project.name}</div>
        <div className="truncate text-[11px] text-faint">{project.path}</div>
      </div>
      <ModelChip />
      <div className="flex-1" />
      {run?.running ? (
        <Button variant="danger" className="h-9.5 px-4 text-sm" onClick={() => void stopRun()}>
          <Square size={14} fill="currentColor" /> Stop my app
        </Button>
      ) : (
        <Button variant="primary" className="h-9.5 px-4 text-sm" onClick={() => void startRun()} disabled={running}>
          <Play size={15} fill="currentColor" /> Run my app
        </Button>
      )}
      <Tooltip content="Make your app into a file you can give to other people">
        <span>
          <Button className="h-9.5 px-3 text-sm" aria-label="Share my app" disabled={running} onClick={() => setConfirmShare(true)}>
            <Package size={15} /> {label('Share my app')}
          </Button>
        </span>
      </Tooltip>
      <Tooltip content={undoTask ? `Put the project back to how it was before “${undoTask.title}”` : 'Nothing to undo yet'}>
        <span>
          <Button className="h-9.5 px-3 text-sm" aria-label="Undo last task" disabled={!undoTask || running} onClick={() => setConfirmUndo(true)}>
            <Undo2 size={15} /> {label('Undo last task')}
          </Button>
        </span>
      </Tooltip>
      <Tooltip content="See every task and go back to before any of them">
        <Button className="h-9.5 px-3 text-sm" aria-label="History" onClick={() => setHistoryOpen(true)}>
          <History size={15} /> {label('History')}
        </Button>
      </Tooltip>
      <Tooltip content={memory ? 'Things I always remember for this project (you have notes)' : 'Things I should always remember for this project'}>
        <Button className={cn('h-9.5 px-3 text-sm', memory && 'border-accent/50')} aria-label="Notes" onClick={() => setNotesOpen(true)}>
          <NotebookPen size={15} /> {label('Notes')}
        </Button>
      </Tooltip>
      <HistoryDialog open={historyOpen} onOpenChange={setHistoryOpen} />
      <NotesDialog open={notesOpen} onOpenChange={setNotesOpen} />
      <Dialog
        open={confirmShare}
        onOpenChange={setConfirmShare}
        title="Share your app?"
        footer={
          <>
            <Button onClick={() => setConfirmShare(false)}>Cancel</Button>
            <Button
              variant="primary"
              onClick={() => {
                setConfirmShare(false)
                void share()
              }}
            >
              <Package size={14} /> Package it
            </Button>
          </>
        }
      >
        <div className="space-y-2 px-5 py-4 text-sm text-muted">
          <p>
            I'll package your app into one file you can give to other people: a program (<b className="text-text">.exe</b>) for desktop apps, or a{' '}
            <b className="text-text">.zip</b> for websites. They won't need to install anything else.
          </p>
          <p>This can take a few minutes, and the first time I may need to download packaging tools. I'll ask before running commands, as usual.</p>
        </div>
      </Dialog>
      <ThemeToggle />
      <IconButton label="Settings" onClick={() => setSettingsOpen(true)}>
        <Settings size={17} />
      </IconButton>
      <Dialog
        open={confirmUndo}
        onOpenChange={setConfirmUndo}
        title="Undo the last task?"
        footer={
          <>
            <Button onClick={() => setConfirmUndo(false)}>Cancel</Button>
            <Button
              variant="primary"
              onClick={() => {
                setConfirmUndo(false)
                void undo()
              }}
            >
              <Undo2 size={14} /> Undo
            </Button>
          </>
        }
      >
        <p className="px-5 py-4 text-sm text-muted">
          This puts your project back to exactly how it was before <b className="text-text">“{undoTask?.title}”</b>. Everything that task added or
          changed will be removed.
        </p>
      </Dialog>
    </header>
  )
}

function ModelChip() {
  const settings = useApp((s) => s.settings)
  const connection = useApp((s) => s.connection)
  const activity = useApp((s) => s.activity)
  const setSettingsOpen = useApp((s) => s.setSettingsOpen)
  const update = useApp((s) => s.updateSettings)
  const running = useWorkspace((s) => s.running)
  const id = activeModelId(settings, connection)
  const model = connection.models.find((m) => m.id === id)
  const connected = connection.status === 'connected'

  let dot = <span className="size-2 shrink-0 rounded-full bg-faint/60" />
  let status = 'Loads when you start'
  if (!connected) {
    dot = <span className="size-2 shrink-0 rounded-full bg-danger" />
    status = 'Not connected'
  } else if (activity === 'loading') {
    dot = <LoaderCircle size={13} className="shrink-0 animate-spin text-accent" />
    status = 'Loading…'
  } else if (model?.loaded) {
    dot = <span className="size-2 shrink-0 rounded-full bg-ok" />
    status = 'Ready'
  }

  const sorted = [...connection.models].sort((a, b) => Number(b.loaded) - Number(a.loaded) || a.id.localeCompare(b.id))
  return (
    <DropdownMenu
      align="start"
      trigger={
        <button
          aria-label="Change the AI model"
          className="ml-2 flex h-9 max-w-[340px] min-w-0 items-center gap-2 rounded-lg border border-border bg-panel-2 px-3 text-left transition-colors hover:border-accent/60"
        >
          {dot}
          <span className="min-w-0 truncate text-[13px]">{id ? modelLabel(id) : 'No model'}</span>
          <span className="shrink-0 text-[11px] text-faint">{status}</span>
          <ChevronDown size={13} className="shrink-0 text-faint" />
        </button>
      }
      items={[
        // Switching mid-task would change models between steps, so it waits for the task to end.
        ...sorted.map((m) => ({
          label: `${modelLabel(m.id)}${m.loaded ? '  (ready)' : ''}`,
          icon: <Check size={13} className={m.id === id ? 'text-accent' : 'invisible'} />,
          disabled: running,
          onSelect: () => void update({ model: m.id })
        })),
        ...(sorted.length ? (['sep'] as const) : []),
        { label: 'Model settings…', icon: <SlidersHorizontal size={13} />, onSelect: () => setSettingsOpen(true) }
      ]}
    />
  )
}

function Conversation() {
  const timeline = useWorkspace((s) => s.timeline)
  const running = useWorkspace((s) => s.running)
  const status = useWorkspace((s) => s.status)
  const ref = useRef<HTMLDivElement>(null)
  const stick = useRef(true)
  const [editing, setEditing] = useState<string | null>(null)

  useLayoutEffect(() => {
    const el = ref.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
  }, [timeline, status, running])

  if (!timeline.length && !running) return <EmptyState />

  // Once thoughts stream in, their own block replaces the "Thinking…" line.
  const last = timeline[timeline.length - 1]
  const thoughtsShown =
    running && status.phase === 'thinking' && last?.kind === 'assistant' && !!(last.reasoning || splitThink(last.text).reasoning)

  return (
    <div
      ref={ref}
      className="min-h-0 flex-1 overflow-y-auto"
      onScroll={(e) => {
        const el = e.currentTarget
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
      }}
    >
      <div className="mx-auto max-w-3xl space-y-5 px-6 py-6">
        {timeline.map((item) => {
          if (item.kind === 'user') {
            if (editing === item.id) return <EditRequest key={item.id} item={item} onClose={() => setEditing(null)} />
            return (
              <div key={item.id} className="group flex flex-col items-end gap-1">
                {item.note && (
                  <span className="flex items-center gap-1 text-[11px] text-faint">
                    <CornerDownRight size={12} />
                    {item.note === 'cut' ? 'Note: stopped my reply to add this' : 'Note added while I was working'}
                  </span>
                )}
                <div className="max-w-[85%] rounded-2xl rounded-br-md bg-accent-soft px-4 py-2.5 text-[14px] whitespace-pre-wrap">{item.text}</div>
                <MessageActions item={item} copyText={item.text} onEdit={() => setEditing(item.id)} />
              </div>
            )
          }
          if (item.kind === 'assistant') {
            const inline = splitThink(item.text)
            const reasoning = [item.reasoning?.trim(), inline.reasoning].filter(Boolean).join('\n\n')
            const live = running && status.phase === 'thinking' && !inline.answer && item.id === timeline[timeline.length - 1].id
            return (
              <div key={item.id} className="group flex gap-3">
                <span className="mt-1 w-6 shrink-0">
                  <LogoMark className="w-6" />
                </span>
                <div className="min-w-0 flex-1 space-y-2">
                  {reasoning && <ThinkingBlock text={reasoning} live={live} />}
                  {inline.answer && <Markdown text={inline.answer} className="text-[14px]" />}
                  {!(running && item.id === last?.id) && <MessageActions item={item} copyText={inline.answer} />}
                </div>
              </div>
            )
          }
          return (
            <div
              key={item.id}
              className={cn(
                'mx-auto max-w-[90%] rounded-lg border px-3.5 py-2 text-center text-[13px]',
                item.tone === 'error' && 'border-danger/30 bg-danger/10 text-danger',
                item.tone === 'warn' && 'border-warn/30 bg-warn/10 text-warn',
                item.tone === 'ok' && 'border-ok/30 bg-ok/10 text-ok',
                item.tone === 'info' && 'border-border bg-panel-2 text-muted'
              )}
            >
              <Markdown text={item.text} />
              {item.reveal && (
                <Button size="sm" className="mt-1.5" onClick={() => void invoke('shell:showItem', item.reveal!)}>
                  <FolderOpen size={13} /> Show in folder
                </Button>
              )}
            </div>
          )
        })}
        {running && status.phase !== 'idle' && !thoughtsShown && (
          <div className="flex items-center gap-2.5 pl-9 text-[13px] text-muted">
            {status.phase === 'waiting' ? (
              <span className="size-2 rounded-full bg-warn" />
            ) : (
              <LoaderCircle size={15} className="animate-spin text-accent" />
            )}
            {status.text}
          </div>
        )}
      </div>
    </div>
  )
}

/**
 * Copy, Delete, and (on the latest request) Edit and Run again. A Delete that can't be used says
 * why when clicked, rather than being greyed out with no reason.
 */
function MessageActions({ item, copyText, onEdit }: { item: TimelineItem; copyText: string; onEdit?: () => void }) {
  const timeline = useWorkspace((s) => s.timeline)
  const tasks = useWorkspace((s) => s.tasks)
  const running = useWorkspace((s) => s.running)
  const deleteItem = useWorkspace((s) => s.deleteItem)
  const redo = useWorkspace((s) => s.redo)
  const [copied, setCopied] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const isUser = item.kind === 'user'
  const noDelete = whyNotDelete(timeline, tasks, item.id, running)
  const canRedo = isUser && !whyNotRedo(timeline, tasks, item.id, running)
  const task = tasks.find((t) => t.id === item.taskId)
  const undoesFiles = !!task?.files.length && !task.undone

  const copy = (): void => {
    void navigator.clipboard.writeText(copyText)
    setCopied(true)
    setTimeout(() => setCopied(false), 1200)
  }

  return (
    <div className={cn('flex items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100', isUser && 'justify-end')}>
      {copyText && (
        <IconButton size="sm" label={copied ? 'Copied' : 'Copy'} onClick={copy}>
          {copied ? <Check size={13} /> : <Copy size={13} />}
        </IconButton>
      )}
      {canRedo && onEdit && (
        <>
          <IconButton size="sm" label={undoesFiles ? 'Edit and run again (undoes this task’s file changes first)' : 'Edit and run again'} onClick={onEdit}>
            <Pencil size={13} />
          </IconButton>
          <IconButton size="sm" label={undoesFiles ? 'Run again (undoes this task’s file changes first)' : 'Run again'} onClick={() => void redo(item.taskId)}>
            <RotateCcw size={13} />
          </IconButton>
        </>
      )}
      <IconButton
        size="sm"
        label={noDelete ?? (isUser ? 'Delete this request and my replies to it' : 'Delete this reply')}
        className={cn(noDelete && 'opacity-40')}
        aria-disabled={!!noDelete}
        onClick={() => (noDelete ? toast(noDelete) : setConfirmDelete(true))}
      >
        <Trash2 size={13} />
      </IconButton>
      <Dialog
        open={confirmDelete}
        onOpenChange={setConfirmDelete}
        title={isUser ? 'Delete this request?' : 'Delete this reply?'}
        footer={
          <>
            <Button onClick={() => setConfirmDelete(false)}>Cancel</Button>
            <Button
              variant="danger"
              onClick={() => {
                setConfirmDelete(false)
                void deleteItem(item.id)
              }}
            >
              <Trash2 size={14} /> Delete
            </Button>
          </>
        }
      >
        <p className="text-sm text-muted">
          {isUser
            ? 'This removes the request, your notes and my replies to it from the conversation, and I forget them. History still lists the task and its steps. This can’t be undone.'
            : 'This removes the reply from the conversation and I forget what it said. The steps I took stay. This can’t be undone.'}
        </p>
      </Dialog>
    </div>
  )
}

/** Edits the latest request in place; "Run again" undoes its file changes first, then runs the new text. */
function EditRequest({ item, onClose }: { item: Extract<TimelineItem, { kind: 'user' }>; onClose: () => void }) {
  const [text, setText] = useState(item.text)
  const ref = useRef<HTMLTextAreaElement>(null)
  const redo = useWorkspace((s) => s.redo)
  const running = useWorkspace((s) => s.running)
  const task = useWorkspace((s) => s.tasks.find((t) => t.id === item.taskId))
  const undoesFiles = !!task?.files.length && !task.undone
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.focus()
    el.setSelectionRange(el.value.length, el.value.length)
  }, [])
  const submit = (): void => {
    if (!text.trim() || running) return
    onClose()
    void redo(item.taskId, text)
  }
  return (
    <div className="ml-auto w-[85%] space-y-2 rounded-xl border border-accent/40 bg-panel p-3">
      <Textarea
        ref={ref}
        value={text}
        rows={Math.min(14, Math.max(3, text.split('\n').length))}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Escape') onClose()
          if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) submit()
        }}
        aria-label="Edit your request"
      />
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] text-faint">
          {undoesFiles ? `Running it again first undoes what “${task!.title}” changed in your files.` : 'Runs as a new task.'}
        </span>
        <div className="flex gap-2">
          <Button size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button size="sm" variant="primary" disabled={!text.trim() || running} onClick={submit}>
            <RotateCcw size={13} /> Run again
          </Button>
        </div>
      </div>
    </div>
  )
}

/** The model's thinking above a reply: open while it streams, folded away once the answer starts (unless the owner keeps it open). */
function ThinkingBlock({ text, live }: { text: string; live: boolean }) {
  const keepOpen = useApp((s) => s.settings?.keepThinkingOpen ?? false)
  const [chosen, setChosen] = useState<boolean | null>(null)
  const [wasLive, setWasLive] = useState(live)
  if (live && !wasLive) setWasLive(true)
  const open = thinkingOpen(chosen, live, wasLive, keepOpen)
  const body = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    if (live && open && body.current) body.current.scrollTop = body.current.scrollHeight
  }, [text, live, open])
  return (
    <div className="rounded-lg border border-border bg-panel-2/60 text-[13px]">
      <button
        type="button"
        className="flex w-full items-center gap-1.5 px-3 py-1.5 text-left text-muted hover:text-text"
        onClick={() => setChosen(!open)}
        aria-expanded={open}
      >
        <ChevronRight size={13} className={cn('shrink-0 transition-transform', open && 'rotate-90')} />
        {live ? <LoaderCircle size={13} className="animate-spin text-accent" /> : <Brain size={13} />}
        {live ? 'Thinking…' : 'Thought process'}
      </button>
      {open && (
        <div ref={body} className="max-h-72 overflow-y-auto border-t border-border px-3 py-2 whitespace-pre-wrap text-muted">
          {text}
        </div>
      )}
    </div>
  )
}

function EmptyState() {
  const project = useWorkspace((s) => s.project)!
  return (
    <div className="flex min-h-0 flex-1 items-center justify-center overflow-y-auto p-8">
      <div className="max-w-lg text-center">
        <LogoMark className="mx-auto w-14" />
        <h2 className="mt-4 text-lg font-semibold">What should I build in “{project.name}”?</h2>
        <p className="mt-2 text-sm text-muted">Describe it in your own words below. Here are some ideas:</p>
        <div className="mt-4 flex flex-col gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              className="rounded-lg border border-border bg-panel px-4 py-2.5 text-left text-sm hover:border-accent/60"
              onClick={() => window.dispatchEvent(new CustomEvent('rulz:fill', { detail: ex }))}
            >
              {ex}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

/** The loop guard or the step limit paused the task. */
function PromptCard() {
  const prompt = useWorkspace((s) => s.prompt)
  const answer = useWorkspace((s) => s.answer)
  const [changing, setChanging] = useState(false)
  const [text, setText] = useState('')
  useEffect(() => {
    setChanging(false)
    setText('')
  }, [prompt?.id])

  if (prompt?.kind === 'question') {
    return (
      <div className="mx-auto w-full max-w-3xl px-6">
        <div className="rounded-lg border border-accent/40 bg-accent-soft/40 px-4 py-2.5 text-[13px]">I asked you a question above. Type your answer below.</div>
      </div>
    )
  }
  if (prompt?.kind !== 'stuck') return null
  const limit = prompt.reason === 'limit'
  return (
    <div className="mx-auto w-full max-w-3xl px-6">
      <div className="rounded-xl border border-warn/40 bg-warn/10 p-4">
        <p className="text-sm">{prompt.message}</p>
        {changing && (
          <Textarea
            className="mt-3"
            rows={2}
            autoFocus
            placeholder="Optional: tell me what to try instead"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
        )}
        <div className="mt-3 flex flex-wrap gap-2">
          <Button variant="primary" onClick={() => answer({ kind: 'stuck', decision: 'continue' })}>
            Keep going
          </Button>
          {!limit &&
            (changing ? (
              <Button onClick={() => answer({ kind: 'stuck', decision: 'change', text })}>Try that</Button>
            ) : (
              <Button onClick={() => setChanging(true)}>Try another way</Button>
            ))}
          <Button variant="ghost" onClick={() => answer({ kind: 'stuck', decision: 'stop' })}>
            Stop
          </Button>
        </div>
      </div>
    </div>
  )
}

function Composer() {
  const running = useWorkspace((s) => s.running)
  const prompt = useWorkspace((s) => s.prompt)
  const send = useWorkspace((s) => s.send)
  const stop = useWorkspace((s) => s.stop)
  const notes = useWorkspace((s) => s.notes)
  const addNote = useWorkspace((s) => s.addNote)
  const cancelNote = useWorkspace((s) => s.cancelNote)
  const interrupt = useWorkspace((s) => s.interrupt)
  const connected = useApp((s) => s.connection.status === 'connected')
  const [text, setText] = useState('')
  const ref = useRef<HTMLTextAreaElement>(null)
  const answering = prompt?.kind === 'question'
  /** While a task runs, what's typed is a note for it (unless it is answering my question). */
  const noting = running && !answering
  const canSend = connected && !!text.trim()

  useEffect(() => {
    const fill = (e: Event): void => {
      setText((e as CustomEvent<string>).detail)
      ref.current?.focus()
    }
    window.addEventListener('rulz:fill', fill)
    return () => window.removeEventListener('rulz:fill', fill)
  }, [])
  useEffect(() => {
    if (answering) ref.current?.focus()
  }, [answering])

  /** `now` cuts the model's current reply short so the note applies at once. */
  const submit = (now = false): void => {
    if (!canSend) return
    if (noting) void addNote(text, now)
    else void send(text)
    setText('')
  }

  return (
    <div className="mx-auto w-full max-w-3xl px-6 pt-3 pb-5">
      {running && notes.length > 0 && (
        <div className="mb-2 flex flex-col gap-1.5">
          {notes.map((n) => (
            <div key={n.id} className="flex items-center gap-2 rounded-lg border border-dashed border-border bg-panel-2 py-1 pr-1 pl-3 text-[13px]">
              <Clock size={14} className="shrink-0 text-faint" />
              <span className="shrink-0 text-faint">Before my next step:</span>
              <span className="min-w-0 flex-1 truncate">{n.text}</span>
              <IconButton size="sm" label="Send now: stop my current reply so this applies at once (running commands keep going)" onClick={interrupt}>
                <CornerDownRight size={14} />
              </IconButton>
              <IconButton size="sm" label="Don't send this note" onClick={() => cancelNote(n.id)}>
                <X size={14} />
              </IconButton>
            </div>
          ))}
        </div>
      )}
      <div className={cn('rounded-xl border bg-panel p-2 transition-colors focus-within:border-accent', answering ? 'border-accent/60' : 'border-border')}>
        <textarea
          ref={ref}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit(e.ctrlKey || e.metaKey)
            }
          }}
          rows={3}
          placeholder={answering ? 'Type your answer…' : running ? 'I’m working on it… add a note and I’ll take it into account before my next step.' : 'What should I build or change?'}
          className="max-h-60 min-h-[76px] w-full resize-none bg-transparent px-2 py-1.5 text-[15px] leading-relaxed placeholder:text-faint focus:outline-none"
          aria-label="What should I build or change?"
        />
        <div className="flex items-center justify-between gap-2 px-1">
          <span className="text-[11px] text-faint">
            {noting ? (
              <>
                <Kbd>Enter</Kbd> add note · <Kbd>Ctrl+Enter</Kbd> send now · <Kbd>Esc</Kbd> to stop
              </>
            ) : (
              <>
                <Kbd>Enter</Kbd> to send · <Kbd>Shift+Enter</Kbd> new line{running && <> · <Kbd>Esc</Kbd> to stop</>}
              </>
            )}
          </span>
          <div className="flex items-center gap-3">
            <DraftMeter />
            <ContextMeter />
            {noting && (
              <>
                <Tooltip content="Stop my current reply and apply this note now (running commands keep going) · Ctrl+Enter">
                  <Button className="h-10 px-3 text-sm" disabled={!canSend} onClick={() => submit(true)} aria-label="Send now">
                    <CornerDownRight size={15} /> Now
                  </Button>
                </Tooltip>
                <Tooltip content="I'll read this before my next step · Enter">
                  <Button variant="primary" className="h-10 px-4 text-sm" disabled={!canSend} onClick={() => submit()}>
                    <MessageSquarePlus size={15} /> Add note
                  </Button>
                </Tooltip>
              </>
            )}
            {running && (
              <Button variant="danger" className="h-10 px-5 text-sm" onClick={stop}>
                <Square size={14} fill="currentColor" /> Stop
              </Button>
            )}
            {(!running || answering) && (
              <Button variant="primary" className="h-10 px-5 text-sm" disabled={!canSend} onClick={() => submit()}>
                <ArrowUp size={16} /> {answering ? 'Answer' : 'Send'}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
