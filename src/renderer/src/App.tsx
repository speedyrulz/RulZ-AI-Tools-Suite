import { Loader2 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Toaster } from 'sonner'
import { InstallDialog } from './components/InstallDialog'
import { SettingsDialog } from './components/SettingsDialog'
import { TopBar } from './components/TopBar'
import { AppPage } from './pages/AppPage'
import { HomePage } from './pages/HomePage'
import { useApps } from './stores/apps'

export function App() {
  const { apps, loaded, selectedId, select, init } = useApps()
  const [query, setQuery] = useState('')
  const [installId, setInstallId] = useState<string | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const scroller = useRef<HTMLDivElement>(null)

  useEffect(() => void init(), [init])
  // Newer Chromium returns a promise from scrollTo; an effect must return nothing.
  useEffect(() => {
    scroller.current?.scrollTo({ top: 0 })
  }, [selectedId])

  const selected = apps.find((a) => a.app.id === selectedId) ?? null
  const installing = apps.find((a) => a.app.id === installId) ?? null

  return (
    <div className="flex h-full flex-col">
      <TopBar
        query={query}
        onQuery={(q) => {
          setQuery(q)
          if (q) select(null)
        }}
        onHome={() => select(null)}
        onSettings={() => setSettingsOpen(true)}
      />
      <div ref={scroller} className="min-h-0 flex-1 overflow-y-auto">
        {!loaded ? (
          <div className="flex h-full items-center justify-center text-faint">
            <Loader2 className="size-6 animate-spin" />
          </div>
        ) : selected ? (
          <AppPage view={selected} onBack={() => select(null)} onInstall={() => setInstallId(selected.app.id)} />
        ) : (
          <HomePage apps={apps} query={query} onOpen={select} onInstall={setInstallId} />
        )}
      </div>

      <InstallDialog view={installing} onClose={() => setInstallId(null)} />
      {settingsOpen && <SettingsDialog onClose={() => setSettingsOpen(false)} />}
      <Toaster
        theme="dark"
        position="bottom-right"
        toastOptions={{ style: { background: '#141417', border: '1px solid #2a2a30', color: '#f2f2f3' } }}
      />
    </div>
  )
}
