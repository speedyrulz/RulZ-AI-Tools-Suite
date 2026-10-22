import { CodeXml, FolderOpen, MessageSquare, Plug, Search, Settings } from 'lucide-react'
import type { ReactNode } from 'react'
import { cn } from '../../lib/format'
import { useApp, type Page } from '../../stores/app'
import { LogoMark } from '../Logo'
import { Tooltip } from '../ui'

const ITEMS: { page: Page; label: string; icon: ReactNode; soon?: boolean }[] = [
  { page: 'chat', label: 'Chat', icon: <MessageSquare size={19} /> },
  { page: 'models', label: 'My Models', icon: <FolderOpen size={19} /> },
  { page: 'mcp', label: 'MCP Tools', icon: <Plug size={19} /> },
  { page: 'discover', label: 'Discover (coming soon)', icon: <Search size={19} />, soon: true },
  { page: 'developer', label: 'Developer (coming soon)', icon: <CodeXml size={19} />, soon: true }
]

export function NavRail() {
  const page = useApp((s) => s.page)
  const setPage = useApp((s) => s.setPage)

  const item = (it: { page: Page; label: string; icon: ReactNode; soon?: boolean }) => (
    <Tooltip key={it.page} content={it.label} side="right">
      <button
        aria-label={it.label}
        onClick={() => setPage(it.page)}
        className={cn(
          'relative flex size-10 items-center justify-center rounded-lg transition-colors',
          page === it.page ? 'bg-accent-soft text-accent' : 'text-muted hover:bg-hover hover:text-text',
          it.soon && page !== it.page && 'opacity-50'
        )}
      >
        {it.icon}
      </button>
    </Tooltip>
  )

  return (
    <nav className="flex w-15 shrink-0 flex-col items-center gap-1.5 border-r border-border bg-panel py-3">
      <div className="mb-3 flex size-10 items-center justify-center select-none">
        <LogoMark className="w-8" />
      </div>
      {ITEMS.map(item)}
      <div className="flex-1" />
      {item({ page: 'settings', label: 'Settings', icon: <Settings size={19} /> })}
    </nav>
  )
}
