import { Moon, Sun } from 'lucide-react'
import { useApp } from '../../stores/app'
import { LogoMark } from '../Logo'
import { IconButton, StatusChip, type StatusTone } from '../ui'

export function ConnectionChip() {
  const status = useApp((s) => s.status)
  const serverState = useApp((s) => s.serverState)
  const mode = useApp((s) => s.settings?.connection.mode ?? 'own')
  const setPage = useApp((s) => s.setPage)

  let tone: StatusTone = 'idle'
  let label = 'Not connected'
  let busy = false
  if (status?.reachable) {
    tone = 'ok'
    label = 'Connected'
  } else if (mode === 'own' && serverState && (serverState.status === 'starting' || serverState.status === 'stopping')) {
    tone = 'warn'
    busy = true
    label = serverState.status === 'starting' ? 'Starting' : 'Stopping'
  } else if (mode === 'own' && serverState?.status === 'error') {
    tone = 'danger'
    label = 'Server error'
  } else if (status === null) {
    busy = true
    label = 'Checking'
  }
  return (
    <button
      onClick={() => setPage('server')}
      className="flex h-8 items-center rounded-md px-2 transition-colors hover:bg-hover"
      aria-label={`Connection: ${label}. Open the Server page`}
    >
      <StatusChip tone={tone} label={label} busy={busy} />
    </button>
  )
}

export function TopBar() {
  const theme = useApp((s) => s.settings?.theme ?? 'dark')
  const setTheme = useApp((s) => s.setTheme)
  const dark = theme === 'dark'
  return (
    <header className="flex h-12 shrink-0 items-center gap-3 border-b border-border bg-panel px-4">
      <div className="flex items-center gap-3 select-none">
        <LogoMark className="h-6 w-auto" />
        <span className="text-[13px] font-semibold tracking-[0.2em] text-muted uppercase">Model Merger</span>
      </div>
      <div className="flex-1" />
      <ConnectionChip />
      <IconButton label={dark ? 'Switch to light mode (Ctrl+Shift+L)' : 'Switch to dark mode (Ctrl+Shift+L)'} onClick={() => void setTheme(dark ? 'light' : 'dark')}>
        {dark ? <Sun size={16} /> : <Moon size={16} />}
      </IconButton>
    </header>
  )
}
