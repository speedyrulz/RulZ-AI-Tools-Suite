import { useEffect } from 'react'
import { Toaster } from 'sonner'
import { NavRail } from './components/shell/NavRail'
import { TopBar } from './components/shell/TopBar'
import { TooltipProvider } from './components/ui'
import { HistoryPage } from './pages/HistoryPage'
import { LoraMergePage } from './pages/LoraMergePage'
import { MergePage } from './pages/MergePage'
import { ResultsPage } from './pages/ResultsPage'
import { ServerPage } from './pages/ServerPage'
import { SettingsPage } from './pages/SettingsPage'
import { SweepPage } from './pages/SweepPage'
import { useApp, type Page } from './stores/app'
import { useGrid } from './stores/grid'
import { useHistory } from './stores/history'
import { useLoraMerge } from './stores/loraMerge'
import { useMerge } from './stores/merge'

function CurrentPage({ page }: { page: Page }) {
  switch (page) {
    case 'merge':
      return <MergePage />
    case 'loras':
      return <LoraMergePage />
    case 'sweep':
      return <SweepPage />
    case 'results':
      return <ResultsPage />
    case 'history':
      return <HistoryPage />
    case 'server':
      return <ServerPage />
    case 'settings':
      return <SettingsPage />
  }
}

export function App() {
  const page = useApp((s) => s.page)
  const theme = useApp((s) => s.settings?.theme ?? 'dark')
  const ready = useApp((s) => s.settings !== null)
  const init = useApp((s) => s.init)
  const setTheme = useApp((s) => s.setTheme)

  useEffect(() => init(), [init])
  // Merge jobs are followed for the whole session, also while another page is open.
  useEffect(() => useMerge.getState().init(), [])
  useEffect(() => useLoraMerge.getState().init(), [])
  useEffect(() => useGrid.getState().init(), [])
  useEffect(() => useHistory.getState().init(), [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'l') {
        e.preventDefault()
        void setTheme(useApp.getState().settings?.theme === 'light' ? 'dark' : 'light')
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [setTheme])

  return (
    <TooltipProvider>
      <div className="flex h-full flex-col bg-bg text-text">
        <TopBar />
        <div className="flex min-h-0 flex-1">
          <NavRail />
          <main className="min-w-0 flex-1 overflow-y-auto">
            <div className="mx-auto max-w-[1280px] p-8">{ready ? <CurrentPage page={page} /> : <p className="text-[13px] text-muted">Loading your settings.</p>}</div>
          </main>
        </div>
      </div>
      <Toaster theme={theme} position="bottom-right" />
    </TooltipProvider>
  )
}
