import { ChevronDown, ExternalLink, Play, Sparkles, Square, Wrench } from 'lucide-react'
import { useLayoutEffect, useRef, useState } from 'react'
import { outputLooksLikeError } from '@shared/runOutput'
import { urlOnComputer } from '../lib/computers'
import { cn } from '../lib/format'
import { openInBrowser } from '../lib/native'
import { currentComputer } from '../stores/link'
import { useWorkspace } from '../stores/workspace'
import { Button, Input } from './ui'

/** "Run my app" on the computer: start and stop it, watch its output, and ask for a fix. */
export function RunView() {
  const run = useWorkspace((s) => s.run)
  const running = useWorkspace((s) => s.running)
  const startRun = useWorkspace((s) => s.startRun)
  const stopRun = useWorkspace((s) => s.stopRun)
  const askHowToRun = useWorkspace((s) => s.askHowToRun)
  const fixRun = useWorkspace((s) => s.fixRun)
  const setRunCommand = useWorkspace((s) => s.setRunCommand)
  const [details, setDetails] = useState(false)
  const [typing, setTyping] = useState(false)
  const [command, setCommand] = useState('')
  const outRef = useRef<HTMLPreElement>(null)

  useLayoutEffect(() => {
    if (outRef.current) outRef.current.scrollTop = outRef.current.scrollHeight
  }, [run?.output])

  const startStop = run?.running ? (
    <Button variant="danger" size="lg" className="w-full" onClick={() => void stopRun()}>
      <Square size={15} fill="currentColor" /> Stop my app
    </Button>
  ) : (
    <Button variant="primary" size="lg" className="w-full" onClick={() => void startRun()} disabled={running}>
      <Play size={16} fill="currentColor" /> Run my app
    </Button>
  )

  if (run?.unknown) {
    return (
      <div className="space-y-3 p-4">
        <p className="text-[15px]">I don't know how to start this app yet.</p>
        {typing ? (
          <div className="space-y-2">
            <Input autoFocus placeholder="PowerShell command, e.g. npm start" value={command} onChange={(e) => setCommand(e.target.value)} />
            <Button
              variant="primary"
              className="w-full"
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
          <div className="flex flex-col gap-2">
            <Button variant="primary" disabled={running} onClick={() => void askHowToRun()}>
              <Sparkles size={15} /> Ask the AI to work it out
            </Button>
            <Button variant="ghost" onClick={() => setTyping(true)}>
              I know the command
            </Button>
          </div>
        )}
      </div>
    )
  }

  if (!run || (!run.running && !run.output && !run.opened)) {
    return (
      <div className="space-y-3 p-4">
        <p className="text-sm text-muted">Start your app on the computer and watch what it prints here.</p>
        {startStop}
      </div>
    )
  }

  const state = run.opened
    ? `Opened ${run.opened} in the computer's web browser.`
    : run.running
      ? 'Your app is running on the computer.'
      : run.stoppedByOwner
        ? 'You stopped your app.'
        : run.exitCode === 0 || run.exitCode === null
          ? 'Your app has closed.'
          : `Your app stopped with an error (code ${run.exitCode}).`
  const crashed = !run.running && !!run.exitCode && !run.stoppedByOwner
  const canFix = !run.opened && (crashed || (run.running && outputLooksLikeError(run.output)))
  const host = currentComputer()?.host

  return (
    <div className="flex min-h-full flex-col gap-3 p-4">
      <div className="flex items-center gap-2">
        <span className={cn('size-2.5 shrink-0 rounded-full', run.running ? 'bg-ok' : crashed ? 'bg-danger' : 'bg-faint')} />
        <span className="text-[15px]">{state}</span>
      </div>
      {startStop}
      <div className="flex flex-wrap gap-2">
        {canFix && (
          <Button variant="primary" disabled={running} onClick={() => void fixRun()}>
            <Wrench size={15} /> Fix this
          </Button>
        )}
        {run.url && host && (
          <Button onClick={() => void openInBrowser(urlOnComputer(run.url!, host))}>
            <ExternalLink size={15} /> Open on this phone
          </Button>
        )}
      </div>
      {run.url && (
        <p className="text-[12px] text-faint">
          If it doesn't open, your app only accepts visits from the computer itself (for Vite, start it with <span className="font-mono">--host</span>).
        </p>
      )}
      {!run.opened && (
        <pre ref={outRef} className="max-h-[55dvh] min-h-24 overflow-auto rounded-lg border border-border bg-code px-3 py-2 font-mono text-[12px] whitespace-pre-wrap text-muted">
          {run.output || 'Starting…'}
        </pre>
      )}
      <div>
        <button className="flex items-center gap-1 py-1 text-[13px] text-faint" onClick={() => setDetails(!details)}>
          <ChevronDown size={14} className={cn('transition-transform', !details && '-rotate-90')} /> Show details
        </button>
        {details && (
          <div className="mt-1 space-y-2">
            <p className="font-mono text-[12px] break-all text-muted">
              {run.source === 'saved' ? 'Saved command' : 'Detected command'}: {run.command}
            </p>
            <Button size="sm" onClick={() => void setRunCommand(null)}>
              Reset how to run
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}
