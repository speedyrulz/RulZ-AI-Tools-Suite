import { Loader2 } from 'lucide-react'
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Toaster } from 'sonner'
import { homeOrder, neighbors } from '@shared/arrange'
import { InstallDialog } from './components/InstallDialog'
import { RequirementGuide } from './components/RequirementGuide'
import { SelfUpdateDialog } from './components/SelfUpdateDialog'
import { ThemeChooser } from './components/ThemeChooser'
import { SettingsDialog } from './components/SettingsDialog'
import { TopBar } from './components/TopBar'
import { AppPage } from './pages/AppPage'
import { HomePage } from './pages/HomePage'
import { useApps } from './stores/apps'

/** True while a dialog or menu is open: its own keys (Esc closes it) come first. */
function overlayOpen(): boolean {
  return !!document.querySelector('[role="dialog"], [role="menu"]')
}

function typingIn(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName))
}

export function App() {
  // One selector per field: the whole store changes with every batch of log lines.
  const apps = useApps((s) => s.apps)
  const loaded = useApps((s) => s.loaded)
  const selectedId = useApps((s) => s.selectedId)
  const select = useApps((s) => s.select)
  const setHomeFilter = useApps((s) => s.setHomeFilter)
  const init = useApps((s) => s.init)
  const theme = useApps((s) => s.theme)
  const needsTheme = useApps((s) => !!s.settings && !s.settings.theme)
  const [query, setQuery] = useState('')
  const [installId, setInstallId] = useState<string | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const scroller = useRef<HTMLDivElement>(null)
  const search = useRef<HTMLInputElement>(null)
  /** Where the home page was scrolled to when an app's page was opened, to come back to the same place. */
  const homeScroll = useRef(0)
  /** The search text, for the handlers that are bound once. */
  const queryNow = useRef(query)
  queryNow.current = query
  /** The app whose page was opened from the home page: its card gets the keyboard focus back afterwards. */
  const opened = useRef<string | null>(null)
  /** Set by Previous / Next, so the same arrow has the keyboard focus on the page it leads to. */
  const arrow = useRef<'previous' | 'next' | null>(null)

  useEffect(() => void init(), [init])

  const selected = apps.find((a) => a.app.id === selectedId) ?? null
  const installing = apps.find((a) => a.app.id === installId) ?? null

  const open = useCallback(
    (id: string): void => {
      if (!useApps.getState().selectedId) homeScroll.current = scroller.current?.scrollTop ?? 0
      opened.current = id
      arrow.current = null
      select(id)
    },
    [select]
  )
  const back = useCallback((): void => select(null), [select])
  const onQuery = useCallback(
    (q: string): void => {
      const starting = !queryNow.current.trim() && !!q.trim()
      setQuery(q)
      if (!q.trim()) return
      // A new search looks through every app; a filter picked after that narrows its results and stays.
      if (starting) setHomeFilter('all')
      // The results change with every letter: show them from the top, on the home page.
      homeScroll.current = 0
      opened.current = null
      select(null)
      scroller.current?.scrollTo({ top: 0 })
    },
    [select, setHomeFilter]
  )

  // An app's page opens at its top; the home page comes back where it was left, with the keyboard focus on the
  // card that was opened. Before paint, so nothing jumps.
  // (Newer Chromium returns a promise from scrollTo; an effect must return nothing.)
  const showing = selected?.app.id ?? null
  useLayoutEffect(() => {
    const el = scroller.current
    if (!el) return
    el.scrollTo({ top: showing ? 0 : homeScroll.current })
    if (showing || !opened.current) return
    el.querySelector<HTMLElement>(`[data-app-id="${CSS.escape(opened.current)}"]`)?.focus({ preventScroll: true })
    opened.current = null
  }, [showing])

  // Previous / Next on an app's page follow the order of the home page.
  const order = useMemo(() => homeOrder(apps), [apps])
  const near = showing ? neighbors(order, showing) : null
  const neighbor = (which: 'previous' | 'next'): { name: string; go: () => void } | null => {
    const view = near && apps.find((a) => a.app.id === near[which])
    if (!view) return null
    return {
      name: view.app.name,
      go: () => {
        arrow.current = which
        select(view.app.id)
      }
    }
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (e.defaultPrevented || overlayOpen()) return
      const key = e.key.toLowerCase()
      const focusSearch = (): void => {
        e.preventDefault()
        search.current?.focus()
        search.current?.select()
      }
      if ((e.ctrlKey && !e.altKey && (key === 'k' || key === 'f')) || (key === '/' && !e.ctrlKey && !e.altKey && !typingIn(e.target))) {
        focusSearch()
      } else if (e.key === 'Escape') {
        // On an app's page: back to the list, search results included. On the list: clear the search, wherever
        // the cursor is; with nothing to clear, leave the search box.
        if (useApps.getState().selectedId) {
          e.preventDefault()
          back()
        } else if (queryNow.current) {
          setQuery('')
        } else if (e.target === search.current) {
          search.current?.blur()
        }
      } else if (e.altKey && e.key === 'ArrowLeft' && useApps.getState().selectedId) {
        e.preventDefault()
        back()
      }
    }
    // The mouse's Back button (button 3) leaves an app's page.
    const onMouse = (e: MouseEvent): void => {
      if (e.button !== 3 || overlayOpen() || !useApps.getState().selectedId) return
      e.preventDefault()
      back()
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('mouseup', onMouse)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('mouseup', onMouse)
    }
  }, [back])

  return (
    <div className="flex h-full flex-col">
      <TopBar query={query} onQuery={onQuery} onHome={back} onSettings={() => setSettingsOpen(true)} searchRef={search} />
      <div ref={scroller} data-scroller className="min-h-0 flex-1 overflow-y-auto [scrollbar-gutter:stable]">
        {!loaded ? (
          <div className="flex h-full items-center justify-center text-faint">
            <Loader2 className="size-6 animate-spin" />
          </div>
        ) : selected ? (
          <AppPage
            // A new page per app: its dialogs, switches in progress and reading position don't carry over.
            key={selected.app.id}
            view={selected}
            onBack={back}
            onInstall={() => setInstallId(selected.app.id)}
            previous={neighbor('previous')}
            next={neighbor('next')}
            focusArrow={arrow.current}
          />
        ) : (
          <HomePage apps={apps} query={query} onQuery={onQuery} onOpen={open} onInstall={setInstallId} />
        )}
      </div>

      <InstallDialog view={installing} onClose={() => setInstallId(null)} />
      {settingsOpen && <SettingsDialog onClose={() => setSettingsOpen(false)} />}
      <RequirementGuide />
      <SelfUpdateDialog />
      {needsTheme && <ThemeChooser />}
      <Toaster
        theme={theme}
        position="bottom-right"
        toastOptions={{ style: { background: 'var(--panel)', border: '1px solid var(--border)', color: 'var(--text)' } }}
      />
    </div>
  )
}
