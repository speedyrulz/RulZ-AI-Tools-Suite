import { Grid3x3, History, Images, Layers, Merge, Server, Settings } from 'lucide-react'
import type { ReactNode } from 'react'
import { cn } from '../../lib/format'
import { useApp, type Page } from '../../stores/app'
import { Tooltip } from '../ui'

interface Item {
  page: Page
  label: string
  icon: ReactNode
}

const TOP: Item[] = [
  { page: 'merge', label: 'Merge', icon: <Merge size={16} /> },
  { page: 'loras', label: 'LoRA merge', icon: <Layers size={16} /> },
  { page: 'sweep', label: 'Sweep', icon: <Grid3x3 size={16} /> },
  { page: 'results', label: 'Results', icon: <Images size={16} /> },
  { page: 'history', label: 'History', icon: <History size={16} /> }
]
const BOTTOM: Item[] = [
  { page: 'server', label: 'Server', icon: <Server size={16} /> },
  { page: 'settings', label: 'Settings', icon: <Settings size={16} /> }
]

export function NavRail() {
  const page = useApp((s) => s.page)
  const setPage = useApp((s) => s.setPage)

  const item = (it: Item) => (
    <Tooltip key={it.page} content={it.label} side="right">
      <button
        aria-label={it.label}
        aria-current={page === it.page ? 'page' : undefined}
        onClick={() => setPage(it.page)}
        className={cn(
          'flex size-10 items-center justify-center rounded-lg transition-colors',
          page === it.page ? 'bg-accent-soft text-accent-text' : 'text-muted hover:bg-hover hover:text-text'
        )}
      >
        {it.icon}
      </button>
    </Tooltip>
  )

  return (
    <nav aria-label="Pages" className="flex w-15 shrink-0 flex-col items-center gap-1.5 border-r border-border bg-panel py-3">
      {TOP.map(item)}
      <div className="flex-1" />
      {BOTTOM.map(item)}
    </nav>
  )
}
