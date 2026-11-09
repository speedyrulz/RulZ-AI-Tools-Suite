import { CloudOff, LoaderCircle } from 'lucide-react'
import type { LinkState } from '../lib/remote'
import { link } from '../lib/remote'
import { useLink } from '../stores/link'
import { Button } from './ui'

/** Why the phone isn't connected, in plain words. */
export function linkProblem(state: LinkState, name: string): { title: string; text: string; retry: boolean } | null {
  switch (state.kind) {
    case 'connected':
    case 'idle':
      return null
    case 'connecting':
      return state.attempt === 0
        ? { title: `Connecting to ${name}…`, text: '', retry: false }
        : {
            title: `Can't reach ${name}`,
            text: 'Check that Tailscale is on, here and on the computer, that RulZ Coding Agent is running there, and that Phone access is on in its Settings. I keep trying.',
            retry: true
          }
    case 'denied':
      return state.reason === 'locked'
        ? { title: 'Too many wrong codes', text: 'The computer stopped listening to this phone for 10 minutes. Then scan its QR code again.', retry: true }
        : {
            title: 'The pairing code doesn’t match',
            text: 'It may have been changed on the computer. Open Settings > Phone access on the computer and scan the QR code again.',
            retry: true
          }
    case 'outdated':
      return state.theirs > state.ours
        ? { title: 'This app needs an update', text: 'RulZ Coding Agent on the computer is newer than this app. Install the latest version of this app.', retry: false }
        : { title: 'The computer needs an update', text: 'This app is newer than RulZ Coding Agent on the computer. Update it there.', retry: false }
  }
}

/** A floating strip above the message box while the connection is down (the page stays as it was). */
export function LinkBanner() {
  const state = useLink((s) => s.state)
  const name = useLink((s) => s.computers.find((c) => c.id === s.currentId)?.name ?? 'your computer')
  if (state.kind === 'connected' || state.kind === 'idle') return null
  const problem = linkProblem(state, name)
  return (
    <div
      role="status"
      className="fixed inset-x-3 bottom-[calc(var(--safe-area-inset-bottom,0px)+76px)] z-30 flex items-center gap-2 rounded-xl border border-warn/40 bg-panel px-3 py-2 text-[13px] text-warn shadow-lg"
    >
      {state.kind === 'connecting' ? <LoaderCircle size={15} className="shrink-0 animate-spin" /> : <CloudOff size={15} className="shrink-0" />}
      <span className="min-w-0 flex-1">{state.kind === 'connecting' ? `Reconnecting to ${name}…` : problem?.title}</span>
      {state.kind === 'connecting' && state.attempt > 1 && (
        <Button size="sm" variant="ghost" className="h-7 text-warn" onClick={() => link.retryNow()}>
          Try now
        </Button>
      )}
    </div>
  )
}
