import { ArrowLeft, ArrowUp, History, LoaderCircle, Play, Settings, Square, Undo2 } from 'lucide-react'
import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { ApprovalDialog } from '../components/ApprovalDialog'
import { HistoryDialog } from '../components/HistoryDialog'
import { LogoMark } from '../components/Logo'
import { Markdown } from '../components/Markdown'
import { ConnectionHelp } from '../components/NotConnected'
import { RunPanel } from '../components/RunPanel'
import { SidePanel } from '../components/SidePanel'
import { modelLabel } from '../components/common'
import { Button, Dialog, IconButton, Kbd, Textarea, Tooltip } from '../components/ui'
import { cn } from '../lib/format'
import { activeModelId, useApp } from '../stores/app'
import { undoableTask, useWorkspace } from '../stores/workspace'
import { ThemeToggle } from './Home'

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
  const setSettingsOpen = useApp((s) => s.setSettingsOpen)
  const [confirmUndo, setConfirmUndo] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(false)
  const undoTask = undoableTask(tasks)

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
      <Tooltip content={undoTask ? `Put the project back to how it was before “${undoTask.title}”` : 'Nothing to undo yet'}>
        <span>
          <Button className="h-9.5 px-4 text-sm" disabled={!undoTask || running} onClick={() => setConfirmUndo(true)}>
            <Undo2 size={15} /> Undo last task
          </Button>
        </span>
      </Tooltip>
      <Tooltip content="See every task and go back to before any of them">
        <Button className="h-9.5 px-4 text-sm" onClick={() => setHistoryOpen(true)}>
          <History size={15} /> History
        </Button>
      </Tooltip>
      <HistoryDialog open={historyOpen} onOpenChange={setHistoryOpen} />
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

  return (
    <Tooltip content="Change the AI model in Settings">
      <button
        onClick={() => setSettingsOpen(true)}
        className="ml-2 flex h-9 max-w-[340px] min-w-0 items-center gap-2 rounded-lg border border-border bg-panel-2 px-3 text-left transition-colors hover:border-accent/60"
      >
        {dot}
        <span className="min-w-0 truncate text-[13px]">{id ? modelLabel(id) : 'No model'}</span>
        <span className="shrink-0 text-[11px] text-faint">{status}</span>
      </button>
    </Tooltip>
  )
}

function Conversation() {
  const timeline = useWorkspace((s) => s.timeline)
  const running = useWorkspace((s) => s.running)
  const status = useWorkspace((s) => s.status)
  const ref = useRef<HTMLDivElement>(null)
  const stick = useRef(true)

  useLayoutEffect(() => {
    const el = ref.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
  }, [timeline, status, running])

  if (!timeline.length && !running) return <EmptyState />

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
            return (
              <div key={item.id} className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-br-md bg-accent-soft px-4 py-2.5 text-[14px] whitespace-pre-wrap">{item.text}</div>
              </div>
            )
          }
          if (item.kind === 'assistant') {
            return (
              <div key={item.id} className="flex gap-3">
                <span className="mt-1 w-6 shrink-0">
                  <LogoMark className="w-6" />
                </span>
                <Markdown text={item.text} className="min-w-0 flex-1 text-[14px]" />
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
            </div>
          )
        })}
        {running && status.phase !== 'idle' && (
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
  const connected = useApp((s) => s.connection.status === 'connected')
  const [text, setText] = useState('')
  const ref = useRef<HTMLTextAreaElement>(null)
  const answering = prompt?.kind === 'question'
  const canSend = connected && !!text.trim() && (!running || answering)

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

  const submit = (): void => {
    if (!canSend) return
    void send(text)
    setText('')
  }

  return (
    <div className="mx-auto w-full max-w-3xl px-6 pt-3 pb-5">
      <div className={cn('rounded-xl border bg-panel p-2 transition-colors focus-within:border-accent', answering ? 'border-accent/60' : 'border-border')}>
        <textarea
          ref={ref}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit()
            }
          }}
          rows={3}
          placeholder={answering ? 'Type your answer…' : running ? 'I’m working on it… you can type your next idea here.' : 'What should I build or change?'}
          className="max-h-60 min-h-[76px] w-full resize-none bg-transparent px-2 py-1.5 text-[15px] leading-relaxed placeholder:text-faint focus:outline-none"
          aria-label="What should I build or change?"
        />
        <div className="flex items-center justify-between gap-2 px-1">
          <span className="text-[11px] text-faint">
            <Kbd>Enter</Kbd> to send · <Kbd>Shift+Enter</Kbd> new line{running && <> · <Kbd>Esc</Kbd> to stop</>}
          </span>
          <div className="flex gap-2">
            {running && (
              <Button variant="danger" className="h-10 px-5 text-sm" onClick={stop}>
                <Square size={14} fill="currentColor" /> Stop
              </Button>
            )}
            {(!running || answering) && (
              <Button variant="primary" className="h-10 px-5 text-sm" disabled={!canSend} onClick={submit}>
                <ArrowUp size={16} /> {answering ? 'Answer' : 'Send'}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
