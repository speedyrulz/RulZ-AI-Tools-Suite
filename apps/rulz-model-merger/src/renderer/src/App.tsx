import { useEffect } from 'react'
import { Toaster } from 'sonner'
import { NavRail } from './components/shell/NavRail'
import { TopBar } from './components/shell/TopBar'
import { TooltipProvider } from './components/ui'
import { EmptyPage } from './pages/EmptyPage'
import { GridPage } from './pages/GridPage'
import { LoraMergePage } from './pages/LoraMergePage'
import { MergePage } from './pages/MergePage'
import { ServerPage } from './pages/ServerPage'
import { SettingsPage } from './pages/SettingsPage'
import { useApp, type Page } from './stores/app'
import { useGrid } from './stores/grid'
import { useLoraMerge } from './stores/loraMerge'
import { useMerge } from './stores/merge'

const EMPTY: Partial<Record<Page, { title: string; text: string }>> = {
  sweep: { title: 'Sweep', text: 'Sweeps come in a later version. Set up the ComfyUI connection on the Server page first.' },
  results: { title: 'Results', text: 'Results come in a later version. Nothing has been rendered yet.' },
  history: { title: 'History', text: 'History comes in a later version. Nothing has been merged yet.' }
}

function CurrentPage({ page }: { page: Page }) {
  if (page === 'merge') return <MergePage />
  if (page === 'loras') return <LoraMergePage />
  if (page === 'grids') return <GridPage />
  if (page === 'server') return <ServerPage />
  if (page === 'settings') return <SettingsPage />
  const e = EMPTY[page]
  return e ? <EmptyPage title={e.title} text={e.text} /> : null
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
