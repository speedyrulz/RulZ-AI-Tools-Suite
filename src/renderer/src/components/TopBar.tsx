import { Moon, RefreshCw, Search, Settings, Sun, X } from 'lucide-react'
import { cn } from '../lib/format'
import { LOGOS } from '../lib/theme'
import { useApps } from '../stores/apps'
import { IconButton } from './ui'

// The window has no native title bar; this bar is the drag area, and Windows draws the
// minimize/maximize/close buttons over its right end (the 150px right padding keeps clear of them).
export function TopBar({
  query,
  onQuery,
  onHome,
  onSettings
}: {
  query: string
  onQuery: (q: string) => void
  onHome: () => void
  onSettings: () => void
}) {
  const refreshing = useApps((s) => s.refreshing)
  const refresh = useApps((s) => s.refresh)
  const theme = useApps((s) => s.theme)
  const setTheme = useApps((s) => s.setTheme)

  return (
    <div className="drag flex h-12 shrink-0 items-center gap-4 border-b border-border bg-bg pr-[150px] pl-4">
      <button onClick={onHome} className="no-drag flex cursor-pointer items-center gap-2.5" title="All apps">
        <img src={LOGOS[theme].mark} alt="RulZ" className="h-6" draggable={false} />
        <span className="font-[family-name:var(--font-display)] text-[13px] font-semibold tracking-[0.2em] text-muted uppercase">
          AI Tools Suite
        </span>
      </button>

      <div className="flex-1" />

      <div className="no-drag relative w-[300px]">
        <Search className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-faint" />
        <input
          value={query}
          onChange={(e) => onQuery(e.target.value)}
          placeholder="Search apps"
          spellCheck={false}
          className="h-8 w-full rounded-lg border border-border bg-panel pr-8 pl-9 text-[13px] outline-none placeholder:text-faint focus:border-accent"
        />
        {query && (
          <button
            onClick={() => onQuery('')}
            className="absolute top-1/2 right-2 -translate-y-1/2 cursor-pointer text-faint hover:text-text"
            aria-label="Clear search"
          >
            <X className="size-4" />
          </button>
        )}
      </div>

      <div className="no-drag flex items-center gap-1">
        <IconButton label="Check for updates" onClick={() => void refresh()} disabled={refreshing}>
          <RefreshCw className={cn('size-4', refreshing && 'animate-spin')} />
        </IconButton>
        <IconButton
          label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          onClick={() => void setTheme(theme === 'dark' ? 'light' : 'dark')}
        >
          {theme === 'dark' ? <Sun className="size-4" /> : <Moon className="size-4" />}
        </IconButton>
        <IconButton label="Settings" onClick={onSettings}>
          <Settings className="size-4" />
        </IconButton>
      </div>
    </div>
  )
}
