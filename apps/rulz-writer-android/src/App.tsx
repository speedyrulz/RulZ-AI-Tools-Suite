import { useEffect, useState } from 'react'
import { App as CapacitorApp } from '@capacitor/app'
import { Toaster } from 'sonner'
import { LinkBanner } from './components/LinkStatus'
import { Button, Sheet } from './components/ui'
import { parsePairingLink, type Computer } from './lib/computers'
import { link } from './lib/remote'
import { isNative, onBackButton, onPairingLink, onResume } from './lib/native'
import { Computers, Connecting } from './pages/Computers'
import { Projects } from './pages/Projects'
import { Settings } from './pages/Settings'
import { leaveProject, Workspace } from './pages/Workspace'
import { useAlerts } from './stores/alerts'
import { isDarkTheme, useApp } from './stores/app'
import { listenForFileChanges, resyncEditors, useEntry, useScene } from './stores/editor'
import { useInterview } from './stores/interview'
import { anyJobRunning, useJobs } from './stores/jobs'
import { useLink } from './stores/link'
import { listenForProjectChanges, useProject } from './stores/project'
import { useUi } from './stores/ui'

let started = false

/** Wires up the connection and the phone's buttons once. */
function start(askToPair: (c: Omit<Computer, 'id'>) => void): void {
  if (started) return
  started = true
  void useApp.getState().init()
  void useLink.getState().load()
  void useAlerts.getState().init()
  void useJobs.getState().init()
  void useInterview.getState().init()
  listenForProjectChanges()
  listenForFileChanges()

  // After every (re)connection: the computer let go of the project with the old connection, so open it again,
  // then send what was typed while away and pick up what happened on the computer.
  link.onState((s) => {
    if (s.kind !== 'connected') return
    useApp.setState({ connectedOnce: true })
    void (async () => {
      const project = useProject.getState().state
      if (project) {
        await useProject.getState().reopen()
        await useJobs.getState().refresh(project.folder)
        resyncEditors()
      }
    })()
  })
  let active = true
  const updateBackoff = (): void => {
    link.backgroundIdle = !active && !anyJobRunning(useJobs.getState().jobs)
  }
  useJobs.subscribe(updateBackoff)
  if (isNative) void CapacitorApp.addListener('appStateChange', ({ isActive }) => { active = isActive; updateBackoff() })
  else document.addEventListener('visibilitychange', () => { active = document.visibilityState === 'visible'; updateBackoff() })
  onResume(() => {
    active = true
    updateBackoff()
    void link.check()
  })
  onPairingLink((url) => {
    // A tapped notification (rulzwriter://open/...): the app is already where the job is.
    if (/^rulzwriter:\/\/open\b/i.test(url)) return
    const parsed = parsePairingLink(url)
    if (parsed) askToPair(parsed)
  })
  onBackButton(() => {
    if (document.querySelector('[role="dialog"]')) {
      // The conflict sheet needs an answer; the back button must not skip it.
      if (useScene.getState().buffer?.conflict || useEntry.getState().buffer?.conflict) return true
      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))
      return true
    }
    const ui = useUi.getState()
    if (ui.view === 'projects') return false
    if (ui.view === 'workspace') {
      if (ui.tab !== 'write') {
        ui.setTab('write')
        return true
      }
      void leaveProject().then((left) => left && useUi.getState().go('projects'))
      return true
    }
    ui.back()
    return true
  })
}

export function App() {
  const loaded = useLink((s) => s.loaded)
  const currentId = useLink((s) => s.currentId)
  const ready = useApp((s) => s.connectedOnce)
  const theme = useApp((s) => s.theme)
  const view = useUi((s) => s.view)
  const project = useProject((s) => s.state)
  const [pairing, setPairing] = useState<Omit<Computer, 'id'> | null>(null)
  const add = useLink((s) => s.add)

  useEffect(() => start(setPairing), [])

  // A different computer starts from a clean slate.
  useEffect(() => {
    useApp.setState({ connectedOnce: false })
    useScene.getState().drop()
    useEntry.getState().drop()
    useProject.setState({ state: null, projects: [], trash: [] })
    useJobs.getState().clear()
    useUi.setState({ view: 'projects', sceneId: null })
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
          This phone will be able to read and change the writing projects in RulZ LLM Suite on <b className="text-text">{pairing?.name}</b> ({pairing?.host}:{pairing?.port}). Only
          continue if you just scanned the QR code on your own computer.
        </p>
      </Sheet>
      <Toaster theme={isDarkTheme(theme) ? 'dark' : 'light'} position="top-center" richColors closeButton offset={{ top: 'calc(var(--safe-area-inset-top, 0px) + 12px)' }} toastOptions={{ style: { whiteSpace: 'pre-wrap' } }} />
    </>
  )
}
