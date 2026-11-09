import { ArrowLeft, ArrowUp, Binary, Brain, Check, ChevronDown, ChevronRight, EllipsisVertical, SlidersHorizontal, History, LoaderCircle, NotebookPen, Package, Play, Settings, Square, Undo2 } from 'lucide-react'
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { splitThink, thinkingOpen } from '@shared/think'
import { modelLabel } from '../components/common'
import { LogoMark } from '../components/Logo'
import { Markdown } from '../components/Markdown'
import { HistorySheet, NotesSheet } from '../components/ProjectSheets'
import { ReSheet, useReOverview, type ReTab } from '../components/ReSheet'
import { RunView } from '../components/RunView'
import { DraftReadout, FileCount, FileList, StepList, TaskPicker } from '../components/TaskViews'
import { Button, IconButton, Sheet, Textarea, TopBar } from '../components/ui'
import { cn, formatTokens } from '../lib/format'
import { activeModelId, useApp } from '../stores/app'
import { useUi, type Tab } from '../stores/ui'
import { undoableTask, useWorkspace } from '../stores/workspace'

const EXAMPLES = ['Make a simple to-do list desktop app', 'Make a calculator website with big buttons', 'Make a video upscaler with a drag-and-drop window']

export function Workspace() {
  const tab = useUi((s) => s.tab)
  const connected = useApp((s) => s.connection.status === 'connected' || s.connection.status === 'checking')

  return (
    <div className="flex h-full flex-col">
      <Header />
      {!connected && <LlmBanner />}
      <Tabs />
      <div className="px-safe flex min-h-0 flex-1 flex-col">
        {tab === 'chat' && (
          <>
            <Conversation />
            <PromptCard />
            <Composer />
          </>
        )}
        {tab !== 'chat' && (
          <div className="pb-safe min-h-0 flex-1 overflow-y-auto overscroll-contain">
            {tab === 'steps' && (
              <>
                <TaskPicker />
                <DraftReadout />
                <StepList />
              </>
            )}
            {tab === 'files' && (
              <>
                <TaskPicker />
                <FileList />
              </>
            )}
            {tab === 'app' && <RunView />}
          </div>
        )}
      </div>
    </div>
  )
}

function Header() {
  const project = useWorkspace((s) => s.project)!
  const go = useUi((s) => s.go)
  const [menu, setMenu] = useState(false)
  return (
    <TopBar>
      <IconButton label="Your projects" onClick={() => go('projects')}>
        <ArrowLeft size={21} />
      </IconButton>
      <div className="min-w-0 flex-1 px-1">
        <div className="truncate text-[16px] leading-tight font-semibold">{project.name}</div>
        <ModelLine />
      </div>
      <IconButton label="More" onClick={() => setMenu(true)}>
        <EllipsisVertical size={21} />
      </IconButton>
      <MoreSheet open={menu} onOpenChange={setMenu} />
    </TopBar>
  )
}

/** Quick switch: the models on the computer, ready ones first, and a way to their settings. */
function ModelSheet({ open, onOpenChange, current }: { open: boolean; onOpenChange: (open: boolean) => void; current: string }) {
  const models = useApp((s) => s.connection.models)
  const update = useApp((s) => s.updateSettings)
  const running = useWorkspace((s) => s.running)
  const go = useUi((s) => s.go)
  const setModelSettingsOpen = useUi((s) => s.setModelSettingsOpen)
  const sorted = [...models].sort((a, b) => Number(b.loaded) - Number(a.loaded) || a.id.localeCompare(b.id))
  return (
    <Sheet open={open} onOpenChange={onOpenChange} title="AI model" description={running ? 'You can switch once the task is done.' : undefined}>
      <div className="pb-2">
        {sorted.map((m) => (
          <button
            key={m.id}
            className="flex w-full items-center gap-3 px-5 py-3.5 text-left text-[16px] active:bg-hover disabled:opacity-40"
            disabled={running}
            onClick={() => {
              onOpenChange(false)
              if (m.id !== current) void update({ model: m.id })
            }}
          >
            <Check size={18} className={m.id === current ? 'text-accent' : 'invisible'} />
            <span className="min-w-0">
              <span className="block truncate">{modelLabel(m.id)}</span>
              {m.loaded && <span className="block text-[12px] text-ok">Ready{m.loadedContext ? `, ${formatTokens(m.loadedContext)} context` : ''}</span>}
            </span>
          </button>
        ))}
        {!sorted.length && <p className="px-5 py-3 text-[14px] text-muted">The computer isn't connected to RulZ LLM Suite, so there are no models to choose from.</p>}
        <button
          className="flex w-full items-center gap-3 border-t border-border px-5 py-3.5 text-left text-[16px] active:bg-hover"
          onClick={() => {
            onOpenChange(false)
            setModelSettingsOpen(true)
            go('settings')
          }}
        >
          <SlidersHorizontal size={18} className="text-muted" />
          Model settings
        </button>
      </div>
    </Sheet>
  )
}

function ModelLine() {
  const [open, setOpen] = useState(false)
  const settings = useApp((s) => s.settings)
  const connection = useApp((s) => s.connection)
  const activity = useApp((s) => s.activity)
  const context = useWorkspace((s) => s.context)
  const id = activeModelId(settings, connection)
  const model = connection.models.find((m) => m.id === id)
  let dot = <span className="size-1.5 shrink-0 rounded-full bg-faint/60" />
  if (connection.status !== 'connected') dot = <span className="size-1.5 shrink-0 rounded-full bg-danger" />
  else if (activity === 'loading') dot = <LoaderCircle size={11} className="shrink-0 animate-spin text-accent" />
  else if (model?.loaded) dot = <span className="size-1.5 shrink-0 rounded-full bg-ok" />
  const pct = context?.size ? Math.round(Math.min(1, context.used / context.size) * 100) : null
  return (
    <>
      <button aria-label="Change the AI model" className="flex max-w-full min-w-0 items-center gap-1.5 text-left text-[12px] text-faint" onClick={() => setOpen(true)}>
        {dot}
        <span className="truncate">{id ? modelLabel(id) : 'No model'}</span>
        <ChevronDown size={12} className="shrink-0" />
        {pct !== null && (
          <span className={cn('shrink-0', pct >= 85 ? 'text-danger' : pct >= 60 && 'text-warn')}>
            · memory {pct}% of {formatTokens(context!.size)}
          </span>
        )}
      </button>
      <ModelSheet open={open} onOpenChange={setOpen} current={id} />
    </>
  )
}

/** The computer can't reach RulZ LLM Suite. */
function LlmBanner() {
  const reconnect = useApp((s) => s.reconnect)
  return (
    <div className="flex items-center gap-2 border-b border-danger/30 bg-danger/10 px-3 py-2 text-[13px] text-danger">
      <span className="min-w-0 flex-1">The computer can't reach RulZ LLM Suite. Start its server there.</span>
      <Button size="sm" variant="ghost" className="h-8 text-danger" onClick={() => void reconnect()}>
        Try again
      </Button>
    </div>
  )
}

function Tabs() {
  const tab = useUi((s) => s.tab)
  const setTab = useUi((s) => s.setTab)
  const running = useWorkspace((s) => s.running)
  const appRunning = useWorkspace((s) => !!s.run?.running)
  const item = (t: Tab, label: ReactNode) => (
    <button
      key={t}
      role="tab"
      aria-selected={tab === t}
      onClick={() => setTab(t)}
      className={cn('flex h-11 flex-1 items-center justify-center gap-1.5 border-b-2 text-[14px]', tab === t ? 'border-accent font-medium text-text' : 'border-transparent text-muted')}
    >
      {label}
    </button>
  )
  return (
    <nav role="tablist" className="px-safe flex shrink-0 border-b border-border bg-panel">
      {item('chat', 'Chat')}
      {item('steps', <>Steps {running && <LoaderCircle size={13} className="animate-spin text-accent" />}</>)}
      {item('files', <>Files <FileCount /></>)}
      {item('app', <>App {appRunning && <span className="size-2 rounded-full bg-ok" />}</>)}
    </nav>
  )
}

function MoreSheet({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const running = useWorkspace((s) => s.running)
  const tasks = useWorkspace((s) => s.tasks)
  const run = useWorkspace((s) => s.run)
  const memory = useWorkspace((s) => s.memory)
  const startRun = useWorkspace((s) => s.startRun)
  const stopRun = useWorkspace((s) => s.stopRun)
  const undo = useWorkspace((s) => s.undo)
  const share = useWorkspace((s) => s.share)
  const go = useUi((s) => s.go)
  const setTab = useUi((s) => s.setTab)
  const [sheet, setSheet] = useState<'undo' | 'share' | 'history' | 'notes' | 're' | null>(null)
  const undoTask = undoableTask(tasks)

  const row = (icon: ReactNode, label: string, onClick: () => void, opts: { disabled?: boolean; hint?: string } = {}) => (
    <button
      className="flex w-full items-center gap-4 px-5 py-3.5 text-left text-[16px] active:bg-hover disabled:opacity-40"
      disabled={opts.disabled}
      onClick={() => {
        onOpenChange(false)
        onClick()
      }}
    >
      <span className="text-muted">{icon}</span>
      <span className="min-w-0">
        <span className="block">{label}</span>
        {opts.hint && <span className="block truncate text-[12px] text-faint">{opts.hint}</span>}
      </span>
    </button>
  )

  return (
    <>
      <Sheet open={open} onOpenChange={onOpenChange} title="Project">
        <div className="pb-2">
          {run?.running
            ? row(<Square size={19} />, 'Stop my app', () => void stopRun())
            : row(<Play size={19} />, 'Run my app', () => {
                setTab('app')
                void startRun()
              }, { disabled: running })}
          {row(<Package size={19} />, 'Share my app', () => setSheet('share'), { disabled: running, hint: 'Make it one file to give to other people' })}
          {row(<Undo2 size={19} />, 'Undo last task', () => setSheet('undo'), { disabled: !undoTask || running, hint: undoTask ? `“${undoTask.title}”` : 'Nothing to undo yet' })}
          {row(<Binary size={19} />, 'Reverse engineering', () => setSheet('re'), { hint: 'Analyse or clone a program, practice tasks, sessions done' })}
          {row(<History size={19} />, 'History', () => setSheet('history'))}
          {row(<NotebookPen size={19} />, 'Notes', () => setSheet('notes'), { hint: memory ? 'You have notes for this project' : 'Things I should always remember' })}
          {row(<Settings size={19} />, 'Settings', () => go('settings'))}
        </div>
      </Sheet>
      <ReSheet open={sheet === 're'} onOpenChange={(o) => setSheet(o ? 're' : null)} />
      <HistorySheet open={sheet === 'history'} onOpenChange={(o) => setSheet(o ? 'history' : null)} />
      <NotesSheet open={sheet === 'notes'} onOpenChange={(o) => setSheet(o ? 'notes' : null)} />
      <Sheet
        open={sheet === 'undo'}
        onOpenChange={(o) => !o && setSheet(null)}
        title="Undo the last task?"
        footer={
          <>
            <Button onClick={() => setSheet(null)}>Cancel</Button>
            <Button
              variant="primary"
              onClick={() => {
                setSheet(null)
                void undo()
              }}
            >
              <Undo2 size={15} /> Undo
            </Button>
          </>
        }
      >
        <p className="px-4 pb-4 text-[15px] text-muted">
          This puts your project back to exactly how it was before <b className="text-text">“{undoTask?.title}”</b>. Everything that task added or changed will be
          removed.
        </p>
      </Sheet>
      <Sheet
        open={sheet === 'share'}
        onOpenChange={(o) => !o && setSheet(null)}
        title="Share your app?"
        footer={
          <>
            <Button onClick={() => setSheet(null)}>Cancel</Button>
            <Button
              variant="primary"
              onClick={() => {
                setSheet(null)
                setTab('chat')
                void share()
              }}
            >
              <Package size={15} /> Package it
            </Button>
          </>
        }
      >
        <div className="space-y-2 px-4 pb-4 text-[15px] text-muted">
          <p>
            I'll package your app into one file on the computer: a program (<b className="text-text">.exe</b>) for desktop apps, or a{' '}
            <b className="text-text">.zip</b> for websites.
          </p>
          <p>This can take a few minutes. I'll ask before running commands, as usual.</p>
        </div>
      </Sheet>
    </>
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

  // Once thoughts stream in, their own box replaces the "Thinking…" line.
  const last = timeline[timeline.length - 1]
  const thoughtsShown =
    running && status.phase === 'thinking' && last?.kind === 'assistant' && !!(last.reasoning || splitThink(last.text).reasoning)

  return (
    <div
      ref={ref}
      className="min-h-0 flex-1 overflow-y-auto overscroll-contain"
      onScroll={(e) => {
        const el = e.currentTarget
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
      }}
    >
      <div className="space-y-4 px-4 py-4">
        {timeline.map((item) => {
          if (item.kind === 'user') {
            return (
              <div key={item.id} className="flex justify-end">
                <div className="max-w-[88%] rounded-2xl rounded-br-md bg-accent-soft px-3.5 py-2.5 text-[15px] whitespace-pre-wrap">{item.text}</div>
              </div>
            )
          }
          if (item.kind === 'assistant') {
            const inline = splitThink(item.text)
            const reasoning = [item.reasoning?.trim(), inline.reasoning].filter(Boolean).join('\n\n')
            if (!reasoning && !inline.answer) return null
            const live = running && status.phase === 'thinking' && !inline.answer && item.id === last.id
            return (
              <div key={item.id} className="flex gap-2.5">
                <span className="mt-1 w-5 shrink-0">
                  <LogoMark className="w-5" />
                </span>
                <div className="min-w-0 flex-1 space-y-2">
                  {reasoning && <ThinkingBox text={reasoning} live={live} />}
                  {inline.answer && <Markdown text={inline.answer} className="text-[15px]" />}
                </div>
              </div>
            )
          }
          return (
            <div
              key={item.id}
              className={cn(
                'rounded-lg border px-3 py-2 text-center text-[13px]',
                item.tone === 'error' && 'border-danger/30 bg-danger/10 text-danger',
                item.tone === 'warn' && 'border-warn/30 bg-warn/10 text-warn',
                item.tone === 'ok' && 'border-ok/30 bg-ok/10 text-ok',
                item.tone === 'info' && 'border-border bg-panel-2 text-muted'
              )}
            >
              <Markdown text={item.text} />
              {item.reveal && <p className="mt-1 font-mono text-[11px] break-all text-faint">{item.reveal}</p>}
            </div>
          )
        })}
        {running && status.phase !== 'idle' && !thoughtsShown && (
          <div className="flex items-center gap-2.5 pl-7 text-[14px] text-muted">
            {status.phase === 'waiting' ? <span className="size-2 rounded-full bg-warn" /> : <LoaderCircle size={16} className="animate-spin text-accent" />}
            {status.text}
          </div>
        )}
      </div>
    </div>
  )
}

/** The model's thinking above a reply: open while it streams, folded away once the answer starts (unless the computer's setting keeps it open). */
function ThinkingBox({ text, live }: { text: string; live: boolean }) {
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
    <div className="rounded-lg border border-border bg-panel-2/60 text-[14px]">
      <button type="button" className="flex min-h-10 w-full items-center gap-2 px-3 py-2 text-left text-muted" onClick={() => setChosen(!open)} aria-expanded={open}>
        <ChevronRight size={15} className={cn('shrink-0 transition-transform', open && 'rotate-90')} />
        {live ? <LoaderCircle size={15} className="animate-spin text-accent" /> : <Brain size={15} />}
        {live ? 'Thinking…' : 'Thought process'}
      </button>
      {open && (
        <div ref={body} className="max-h-72 overflow-y-auto overscroll-contain border-t border-border px-3 py-2 whitespace-pre-wrap text-muted">
          {text}
        </div>
      )}
    </div>
  )
}

function EmptyState() {
  const project = useWorkspace((s) => s.project)!
  const connected = useApp((s) => s.connection.status === 'connected')
  const re = useReOverview(connected)
  const [reOpen, setReOpen] = useState<ReTab | null>(null)
  const done = re.data?.sessions.length ?? 0
  const solved = re.data?.sessions.filter((s) => s.outcome === 'solved').length ?? 0
  const practice = re.data?.practice.tasks.length ?? 0
  const unsolved = re.data?.practice.tasks.filter((t) => t.solved === 0).length ?? 0
  return (
    <div className="flex min-h-0 flex-1 items-center justify-center overflow-y-auto p-6">
      <div className="max-w-md text-center">
        <LogoMark className="mx-auto w-12" />
        <h2 className="mt-3 text-[17px] font-semibold">What should I build in “{project.name}”?</h2>
        <p className="mt-1.5 text-[14px] text-muted">Describe it in your own words below. Some ideas:</p>
        <div className="mt-3 flex flex-col gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              className="rounded-xl border border-border bg-panel px-4 py-3 text-left text-[14px] active:border-accent/60"
              onClick={() => window.dispatchEvent(new CustomEvent('rulz:fill', { detail: ex }))}
            >
              {ex}
            </button>
          ))}
        </div>
        {re.data && (
          <div className="mt-5 rounded-xl border border-border bg-panel p-4 text-left">
            <div className="flex items-center gap-2 text-[15px] font-semibold">
              <Binary size={16} className="text-accent" /> Or reverse-engineer a program
            </div>
            <div className="mt-1 text-[13px] text-muted">
              {done} session{done === 1 ? '' : 's'} done ({solved} solved) · {practice} practice task{practice === 1 ? '' : 's'} ({unsolved} not solved)
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <Button size="sm" onClick={() => setReOpen('start')}>
                Analyse or clone
              </Button>
              <Button size="sm" onClick={() => setReOpen('practice')} disabled={!practice}>
                Practice
              </Button>
              <Button size="sm" onClick={() => setReOpen('sessions')} disabled={!done}>
                Done
              </Button>
            </div>
          </div>
        )}
        <ReSheet open={!!reOpen} onOpenChange={(o) => !o && setReOpen(null)} initialTab={reOpen ?? 'start'} />
      </div>
    </div>
  )
}

/** The agent asked a question, or the loop guard or step limit paused the task. */
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
    return <div className="mx-4 rounded-lg border border-accent/40 bg-accent-soft/40 px-3.5 py-2.5 text-[14px]">I asked you a question above. Type your answer below.</div>
  }
  if (prompt?.kind !== 'stuck') return null
  const limit = prompt.reason === 'limit'
  return (
    <div className="mx-4 rounded-xl border border-warn/40 bg-warn/10 p-3.5">
      <p className="text-[15px]">{prompt.message}</p>
      {changing && (
        <Textarea className="mt-3" rows={2} autoFocus placeholder="Optional: tell me what to try instead" value={text} onChange={(e) => setText(e.target.value)} />
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
  )
}

function Composer() {
  const running = useWorkspace((s) => s.running)
  const prompt = useWorkspace((s) => s.prompt)
  const send = useWorkspace((s) => s.send)
  const stop = useWorkspace((s) => s.stop)
  const llmReady = useApp((s) => s.connection.status === 'connected')
  const [text, setText] = useState('')
  const ref = useRef<HTMLTextAreaElement>(null)
  const answering = prompt?.kind === 'question'
  const canSend = llmReady && !!text.trim() && (!running || answering)

  useEffect(() => {
    const fill = (e: Event): void => {
      setText((e as CustomEvent<string>).detail)
      ref.current?.focus()
    }
    window.addEventListener('rulz:fill', fill)
    return () => window.removeEventListener('rulz:fill', fill)
  }, [])
  // Grows with the text, up to about six lines.
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 168)}px`
  }, [text])

  const submit = async (): Promise<void> => {
    if (!canSend) return
    const sent = text
    setText('')
    if (!(await send(sent))) setText(sent)
  }

  return (
    <div className="pb-safe shrink-0 border-t border-border bg-panel px-3 pt-2.5">
      <div className="flex items-end gap-2 pb-2.5">
        <textarea
          ref={ref}
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={1}
          placeholder={answering ? 'Type your answer…' : running ? 'I’m working on it…' : 'What should I build or change?'}
          className={cn(
            'max-h-42 min-h-11 flex-1 resize-none rounded-2xl border bg-panel-2 px-3.5 py-2.5 text-[16px] leading-snug placeholder:text-faint focus:outline-none',
            answering ? 'border-accent/60' : 'border-border focus:border-accent'
          )}
          aria-label="What should I build or change?"
        />
        {running && !answering ? (
          <Button variant="danger" className="size-11 shrink-0 rounded-full px-0" aria-label="Stop" onClick={stop}>
            <Square size={16} fill="currentColor" />
          </Button>
        ) : (
          <Button variant="primary" className="size-11 shrink-0 rounded-full px-0" aria-label={answering ? 'Answer' : 'Send'} disabled={!canSend} onClick={() => void submit()}>
            <ArrowUp size={20} />
          </Button>
        )}
      </div>
    </div>
  )
}
