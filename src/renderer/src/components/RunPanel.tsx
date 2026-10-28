import { ChevronDown, ExternalLink, Sparkles, Square, Wrench, X } from 'lucide-react'
import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { outputLooksLikeError } from '@shared/runOutput'
import { invoke } from '../lib/api'
import { cn } from '../lib/format'
import { useWorkspace } from '../stores/workspace'
import { Button, IconButton, Input } from './ui'

/** Output of "Run my app", or help when the app doesn't know how to run the project. */
export function RunPanel() {
  const run = useWorkspace((s) => s.run)
  const running = useWorkspace((s) => s.running)
  const stopRun = useWorkspace((s) => s.stopRun)
  const askHowToRun = useWorkspace((s) => s.askHowToRun)
  const fixRun = useWorkspace((s) => s.fixRun)
  const setRunCommand = useWorkspace((s) => s.setRunCommand)
  const startRun = useWorkspace((s) => s.startRun)
  const [hidden, setHidden] = useState(false)
  const [details, setDetails] = useState(false)
  const [typing, setTyping] = useState(false)
  const [command, setCommand] = useState('')
  const outRef = useRef<HTMLPreElement>(null)

  useEffect(() => {
    if (run?.running || run?.opened || run?.unknown) setHidden(false)
  }, [run?.running, run?.opened, run?.unknown])
  useLayoutEffect(() => {
    if (outRef.current) outRef.current.scrollTop = outRef.current.scrollHeight
  }, [run?.output])

  if (!run || hidden) return null
  const nothingYet = !!run.unknown
  if (!nothingYet && !run.running && !run.output && !run.opened) return null

  if (nothingYet) {
    return (
      <div className="mx-auto mb-4 w-full max-w-3xl px-6">
        <div className="rounded-xl border border-border bg-panel p-4">
          <div className="flex items-start justify-between gap-3">
            <p className="text-sm">I don't know how to start this app yet.</p>
            <IconButton size="sm" label="Close" onClick={() => setHidden(true)}>
              <X size={14} />
            </IconButton>
          </div>
          {typing ? (
            <div className="mt-3 flex gap-2">
              <Input autoFocus placeholder="PowerShell command, e.g. npm start" value={command} onChange={(e) => setCommand(e.target.value)} />
              <Button
                variant="primary"
                disabled={!command.trim()}
                onClick={async () => {
                  await setRunCommand(command)
                  setTyping(false)
                  await startRun()
                }}
              >
                Save and run
              </Button>
            </div>
          ) : (
            <div className="mt-3 flex gap-2">
              <Button variant="primary" disabled={running} onClick={() => void askHowToRun()}>
                <Sparkles size={14} /> Ask the AI to work it out
              </Button>
              <Button variant="ghost" onClick={() => setTyping(true)}>
                I know the command
              </Button>
            </div>
          )}
        </div>
      </div>
    )
  }

  const state = run.opened
    ? `Opened ${run.opened} in your web browser.`
    : run.running
      ? 'Your app is running.'
      : run.stoppedByOwner
        ? 'You stopped your app.'
        : run.exitCode === 0 || run.exitCode === null
          ? 'Your app has closed.'
        : `Your app stopped with an error (code ${run.exitCode}).`

  const crashed = !run.running && !!run.exitCode && !run.stoppedByOwner
  const canFix = !run.opened && (crashed || (run.running && outputLooksLikeError(run.output)))

  return (
    <div className="mx-auto mb-4 w-full max-w-3xl px-6">
      <div className="overflow-hidden rounded-xl border border-border bg-panel">
        <div className="flex items-center gap-2 border-b border-border px-3 py-2">
          <span className={cn('size-2 rounded-full', run.running ? 'bg-ok' : run.exitCode && !run.stoppedByOwner ? 'bg-danger' : 'bg-faint')} />
          <span className="min-w-0 flex-1 truncate text-[13px]">{state}</span>
          {canFix && (
            <Button
              size="sm"
              variant="primary"
              disabled={running}
              title={running ? 'Wait for the current task to finish' : 'Ask the AI to find and fix the error'}
              onClick={() => void fixRun()}
            >
              <Wrench size={13} /> Fix this
            </Button>
          )}
          {run.url && (
            <Button size="sm" onClick={() => void invoke('shell:openUrl', run.url!)}>
              <ExternalLink size={13} /> Open in browser
            </Button>
          )}
          {run.running && (
            <Button size="sm" variant="danger" onClick={() => void stopRun()}>
              <Square size={11} fill="currentColor" /> Stop app
            </Button>
          )}
          <IconButton size="sm" label="Hide" onClick={() => setHidden(true)}>
            <X size={14} />
          </IconButton>
        </div>
        {!run.opened && (
          <pre ref={outRef} className="max-h-40 min-h-12 overflow-auto bg-code px-3 py-2 font-mono text-[11.5px] whitespace-pre-wrap text-muted">
            {run.output || 'Starting…'}
          </pre>
        )}
        <div className="flex items-center justify-between border-t border-border px-3 py-1.5">
          <button className="flex items-center gap-1 text-[12px] text-faint hover:text-text" onClick={() => setDetails(!details)}>
            <ChevronDown size={13} className={cn('transition-transform', !details && '-rotate-90')} /> Show details
          </button>
          {details && (
            <button
              className="text-[12px] text-faint hover:text-text"
              onClick={() => void setRunCommand(null)}
              title="Forget the saved command and work it out again next time"
            >
              Reset how to run
            </button>
          )}
        </div>
        {details && (
          <p className="px-3 pb-2 font-mono text-[11.5px] text-muted">
            {run.source === 'saved' ? 'Saved command' : 'Detected command'}: {run.command}
          </p>
        )}
      </div>
    </div>
  )
}
