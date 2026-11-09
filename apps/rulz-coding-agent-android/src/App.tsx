import { useEffect, useState } from 'react'
import { Toaster } from 'sonner'
import { ApprovalSheet } from './components/ApprovalSheet'
import { LinkBanner } from './components/LinkStatus'
import { Button, Sheet } from './components/ui'
import { parsePairingLink, type Computer } from './lib/computers'
import { link } from './lib/remote'
import { onBackButton, onPairingLink, onResume } from './lib/native'
import { Computers, Connecting } from './pages/Computers'
import { Projects } from './pages/Projects'
import { Settings } from './pages/Settings'
import { Workspace } from './pages/Workspace'
import { useAlerts } from './stores/alerts'
import { isDarkTheme, useApp } from './stores/app'
import { useLink } from './stores/link'
import { useUi } from './stores/ui'
import { useWorkspace } from './stores/workspace'

let started = false

/** Wires up the connection and the phone's buttons once. */
function start(askToPair: (c: Omit<Computer, 'id'>) => void): void {
  if (started) return
  started = true
  void useApp.getState().init()
  void useLink.getState().load()
  void useAlerts.getState().init()

  // After every (re)connection, load everything again: things may have changed while away.
  link.onState((s) => {
    if (s.kind !== 'connected') return
    void useApp.getState().resync()
    void useWorkspace.getState().resync()
  })
  onResume(() => void link.check())
  onPairingLink((url) => {
    // A tapped notification (rulzagent://open/project): go to the project; a pending approval shows itself.
    if (/^rulzagent:\/\/open\b/i.test(url)) {
      if (useWorkspace.getState().project) {
        useUi.getState().go('workspace')
        useUi.getState().setTab('chat')
      }
      return
    }
    const parsed = parsePairingLink(url)
    if (parsed) askToPair(parsed)
  })
  onBackButton(() => {
    // An unanswered approval stays up; the back button leaves the app instead of refusing by accident.
    if (document.querySelector('[role="dialog"]')) {
      if (useWorkspace.getState().prompt?.kind === 'approval') return false
      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))
      return true
    }
    const ui = useUi.getState()
    if (ui.view === 'projects') return false
    ui.back()
    return true
  })
}

export function App() {
  const loaded = useLink((s) => s.loaded)
  const currentId = useLink((s) => s.currentId)
  const ready = useApp((s) => !!s.settings)
  const theme = useApp((s) => s.theme)
  const view = useUi((s) => s.view)
  const project = useWorkspace((s) => s.project)
  const [pairing, setPairing] = useState<Omit<Computer, 'id'> | null>(null)
  const add = useLink((s) => s.add)

  useEffect(() => start(setPairing), [])

  // A different computer starts from a clean slate.
  useEffect(() => {
    useApp.setState({ settings: null, info: null })
    useWorkspace.setState({ project: null, timeline: [], tasks: [], running: false, prompt: null, run: null })
    useUi.setState({ view: 'projects' })
  }, [currentId])

  let page
  if (!loaded) page = <div className="flex h-full items-center justify-center text-sm text-faint">Starting…</div>
  else if (!currentId) page = <Computers />
  else if (!ready) page = <Connecting />
  else if (view === 'settings') page = <Settings />
  else if (view === 'workspace' && project) page = <Workspace />
  else page = <Projects />

  return (
    <>
      {page}
      {ready && currentId && <LinkBanner />}
      {ready && currentId && <ApprovalSheet />}
      <Sheet
        open={!!pairing}
        onOpenChange={(o) => !o && setPairing(null)}
        title={`Connect to ${pairing?.name}?`}
        footer={
          <>
            <Button onClick={() => setPairing(null)}>Cancel</Button>
            <Button
              variant="primary"
              onClick={() => {
                if (pairing) void add(pairing)
                setPairing(null)
              }}
            >
              Connect
            </Button>
          </>
        }
      >
        <p className="px-4 pb-4 text-[15px] text-muted">
          This phone will be able to see and control RulZ Coding Agent on <b className="text-text">{pairing?.name}</b> ({pairing?.host}:{pairing?.port}). Only
          continue if you just scanned the QR code on your own computer.
        </p>
      </Sheet>
      <Toaster theme={isDarkTheme(theme) ? 'dark' : 'light'} position="top-center" richColors closeButton offset={{ top: 'calc(var(--safe-area-inset-top, 0px) + 12px)' }} toastOptions={{ style: { whiteSpace: 'pre-wrap' } }} />
    </>
  )
}
