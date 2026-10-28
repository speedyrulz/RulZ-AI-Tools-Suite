import { useEffect, useState } from 'react'
import { Toaster } from 'sonner'
import { NotConnectedScreen } from './components/NotConnected'
import { SettingsDialog } from './components/SettingsDialog'
import { TooltipProvider } from './components/ui'
import { Home } from './pages/Home'
import { Wizard } from './pages/Wizard'
import { Workspace } from './pages/Workspace'
import { useApp } from './stores/app'
import { useWorkspace } from './stores/workspace'

export function App() {
  const settings = useApp((s) => s.settings)
  const connection = useApp((s) => s.connection)
  const project = useWorkspace((s) => s.project)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    void useApp
      .getState()
      .init()
      .then(() => setReady(true))
  }, [])

  if (!ready || !settings) {
    return <div className="flex h-full items-center justify-center text-sm text-faint">Starting RulZ Coding Agent…</div>
  }

  let page
  if (!settings.firstRunDone) page = <Wizard />
  else if (project) page = <Workspace />
  else if (connection.status !== 'connected' && connection.status !== 'checking') page = <NotConnectedScreen />
  else page = <Home />

  return (
    <TooltipProvider>
      {page}
      <SettingsDialog />
      <Toaster
        theme={settings.theme === 'light' ? 'light' : settings.theme === 'system' ? 'system' : 'dark'}
        position="bottom-right"
        richColors
        closeButton
        toastOptions={{ style: { whiteSpace: 'pre-wrap' } }}
      />
    </TooltipProvider>
  )
}
