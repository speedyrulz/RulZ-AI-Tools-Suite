import { FolderOpen, FolderPlus, Moon, Settings, Sun, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { ProjectInfo } from '@shared/types'
import { LogoWordmark } from '../components/Logo'
import { Button, IconButton, Input } from '../components/ui'
import { invoke } from '../lib/api'
import { relativeTime } from '../lib/format'
import { isDarkTheme, useApp } from '../stores/app'
import { useWorkspace } from '../stores/workspace'

export function ThemeToggle() {
  const theme = useApp((s) => s.settings?.theme ?? 'dark')
  const setTheme = useApp((s) => s.setTheme)
  const dark = isDarkTheme(theme)
  return (
    <IconButton label={dark ? 'Switch to light mode' : 'Switch to dark mode'} onClick={() => void setTheme(dark ? 'light' : 'dark')}>
      {dark ? <Sun size={17} /> : <Moon size={17} />}
    </IconButton>
  )
}

export function Home() {
  const settings = useApp((s) => s.settings)!
  const setSettingsOpen = useApp((s) => s.setSettingsOpen)
  const open = useWorkspace((s) => s.open)
  const create = useWorkspace((s) => s.create)
  const [recent, setRecent] = useState<ProjectInfo[]>([])
  const [name, setName] = useState('')
  const [parent, setParent] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    void invoke('projects:recent').then(setRecent)
  }, [])

  const folder = `${(parent ?? settings.projectsDir).replace(/[\\/]+$/, '')}\\${name.trim() || '…'}`
  const createProject = async (): Promise<void> => {
    if (!name.trim()) return
    setBusy(true)
    await create(name, parent)
    setBusy(false)
  }

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-14 shrink-0 items-center justify-end gap-1 border-b border-border bg-panel px-4">
        <ThemeToggle />
        <IconButton label="Settings" onClick={() => setSettingsOpen(true)}>
          <Settings size={17} />
        </IconButton>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-2xl space-y-8 px-6 py-10">
          <div className="text-center">
            <LogoWordmark className="mx-auto h-14" />
            <p className="mt-3 text-sm text-muted">Describe an app in plain English and I'll build it for you.</p>
          </div>

          <section className="rounded-xl border border-border bg-panel p-6">
            <h2 className="flex items-center gap-2 text-[15px] font-semibold">
              <FolderPlus size={18} className="text-accent" /> New project
            </h2>
            <div className="mt-4 flex gap-2">
              <Input
                placeholder="Project name, e.g. Video Upscaler"
                value={name}
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && void createProject()}
                className="h-10 text-sm"
              />
              <Button variant="primary" className="h-10 px-5 text-sm" disabled={!name.trim() || busy} onClick={() => void createProject()}>
                Create
              </Button>
            </div>
            <div className="mt-2 flex items-center justify-between gap-3 text-xs text-faint">
              <span className="truncate">Folder: {folder}</span>
              <Button
                size="sm"
                variant="ghost"
                onClick={async () => {
                  const dir = await invoke('dialog:pickFolder', { title: 'Where should the project folder go?', defaultPath: parent ?? settings.projectsDir })
                  if (dir) setParent(dir)
                }}
              >
                Choose folder
              </Button>
            </div>
          </section>

          <section className="rounded-xl border border-border bg-panel p-6">
            <div className="flex items-center justify-between">
              <h2 className="text-[15px] font-semibold">Your projects</h2>
              <Button
                onClick={async () => {
                  const dir = await invoke('dialog:pickFolder', { title: 'Open a project folder', defaultPath: settings.projectsDir })
                  if (dir) await open(dir)
                }}
              >
                <FolderOpen size={15} /> Open existing folder
              </Button>
            </div>
            {recent.length === 0 ? (
              <p className="mt-4 text-sm text-faint">No projects yet. Create one above.</p>
            ) : (
              <ul className="mt-3 divide-y divide-border">
                {recent.map((p) => (
                  <li key={p.path} className="group flex items-center gap-3 py-1">
                    <button className="flex min-w-0 flex-1 items-center gap-3 rounded-md px-2 py-2 text-left hover:bg-hover" onClick={() => void open(p.path)}>
                      <FolderOpen size={17} className="shrink-0 text-accent" />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-sm font-medium">{p.name}</span>
                        <span className="block truncate text-xs text-faint">{p.path}</span>
                      </span>
                      <span className="shrink-0 text-xs text-faint">{relativeTime(p.lastOpened)}</span>
                    </button>
                    <IconButton
                      size="sm"
                      label="Remove from this list (the folder is kept)"
                      className="opacity-0 group-hover:opacity-100"
                      onClick={async () => setRecent(await invoke('projects:forget', p.path))}
                    >
                      <X size={14} />
                    </IconButton>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      </div>
    </div>
  )
}
