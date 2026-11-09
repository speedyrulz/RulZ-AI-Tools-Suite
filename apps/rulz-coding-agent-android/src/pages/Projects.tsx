import { ChevronRight, FolderOpen, FolderPlus, Laptop, LoaderCircle, Settings, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { ProjectInfo } from '@shared/types'
import { FolderPicker } from '../components/FolderPicker'
import { LogoMark } from '../components/Logo'
import { Button, IconButton, Input, TopBar } from '../components/ui'
import { invoke } from '../lib/api'
import { relativeTime } from '../lib/format'
import { useApp } from '../stores/app'
import { useLink } from '../stores/link'
import { useUi } from '../stores/ui'
import { useWorkspace } from '../stores/workspace'

export function Projects() {
  const settings = useApp((s) => s.settings)!
  const computer = useLink((s) => s.computers.find((c) => c.id === s.currentId))
  const select = useLink((s) => s.select)
  const go = useUi((s) => s.go)
  const current = useWorkspace((s) => s.project)
  const running = useWorkspace((s) => s.running)
  const open = useWorkspace((s) => s.open)
  const create = useWorkspace((s) => s.create)
  const [recent, setRecent] = useState<ProjectInfo[]>([])
  const [name, setName] = useState('')
  const [parent, setParent] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [picker, setPicker] = useState<'open' | 'parent' | null>(null)

  useEffect(() => {
    void invoke('projects:recent').then(setRecent, () => undefined)
  }, [current?.path])

  const folder = `${(parent ?? settings.projectsDir).replace(/[\\/]+$/, '')}\\${name.trim() || '…'}`
  const createProject = async (): Promise<void> => {
    if (!name.trim()) return
    setBusy(true)
    if (await create(name, parent)) setName('')
    setBusy(false)
  }

  return (
    <div className="flex h-full flex-col">
      <TopBar>
        <button className="flex min-w-0 flex-1 items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-left active:bg-hover" onClick={() => select(null)} aria-label="Your computers">
          <LogoMark className="w-7 shrink-0" />
          <span className="min-w-0">
            <span className="block truncate text-[16px] font-semibold">{computer?.name}</span>
            <span className="flex items-center gap-1 text-[12px] text-ok">
              <Laptop size={12} /> Connected
            </span>
          </span>
        </button>
        <IconButton label="Settings" onClick={() => go('settings')}>
          <Settings size={20} />
        </IconButton>
      </TopBar>

      <div className="px-safe pb-safe min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-lg space-y-5 px-4 py-5">
          {current && (
            <button className="flex w-full items-center gap-3 rounded-2xl border border-accent/50 bg-accent-soft/40 p-4 text-left active:bg-accent-soft" onClick={() => go('workspace')}>
              {running ? <LoaderCircle size={22} className="shrink-0 animate-spin text-accent" /> : <FolderOpen size={22} className="shrink-0 text-accent" />}
              <span className="min-w-0 flex-1">
                <span className="block text-[12px] text-muted">{running ? 'Working on it now' : 'Open on the computer now'}</span>
                <span className="block truncate text-[17px] font-semibold">{current.name}</span>
              </span>
              <ChevronRight size={20} className="shrink-0 text-muted" />
            </button>
          )}

          <section className="rounded-2xl border border-border bg-panel p-4">
            <h2 className="flex items-center gap-2 text-[16px] font-semibold">
              <FolderPlus size={19} className="text-accent" /> New project
            </h2>
            <div className="mt-3 flex gap-2">
              <Input
                placeholder="Project name, e.g. Video Upscaler"
                value={name}
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && void createProject()}
                enterKeyHint="go"
              />
              <Button variant="primary" disabled={!name.trim() || busy} onClick={() => void createProject()}>
                Create
              </Button>
            </div>
            <div className="mt-2 flex items-center justify-between gap-2 text-[12px] text-faint">
              <span className="min-w-0 break-all">Folder: {folder}</span>
              <Button size="sm" variant="ghost" onClick={() => setPicker('parent')}>
                Change
              </Button>
            </div>
          </section>

          <section className="rounded-2xl border border-border bg-panel p-2">
            <div className="flex items-center justify-between px-2 pt-1">
              <h2 className="text-[16px] font-semibold">Your projects</h2>
              <Button size="sm" onClick={() => setPicker('open')}>
                <FolderOpen size={15} /> Open folder
              </Button>
            </div>
            {recent.length === 0 ? (
              <p className="px-2 py-4 text-sm text-faint">No projects yet. Create one above.</p>
            ) : (
              <ul className="mt-1">
                {recent.map((p) => (
                  <li key={p.path} className="flex items-center">
                    <button className="flex min-w-0 flex-1 items-center gap-3 rounded-xl px-2 py-3 text-left active:bg-hover" onClick={() => void open(p.path)}>
                      <FolderOpen size={19} className="shrink-0 text-accent" />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-[15px] font-medium">{p.name}</span>
                        <span className="block truncate text-[12px] text-faint">
                          {relativeTime(p.lastOpened)} · {p.path}
                        </span>
                      </span>
                    </button>
                    <IconButton label="Remove from this list (the folder is kept)" onClick={async () => setRecent(await invoke('projects:forget', p.path))}>
                      <X size={16} />
                    </IconButton>
                  </li>
                ))}
              </ul>
            )}
          </section>
          <p className="px-2 text-center text-[12px] text-faint">Opening a project here opens it on the computer too.</p>
        </div>
      </div>

      <FolderPicker
        open={picker !== null}
        onOpenChange={(o) => !o && setPicker(null)}
        title={picker === 'open' ? 'Open a project folder' : 'Where should the project folder go?'}
        start={picker === 'parent' ? (parent ?? settings.projectsDir) : settings.projectsDir}
        action={picker === 'open' ? 'Open this folder' : 'Use this folder'}
        onPick={(path) => {
          if (picker === 'open') void open(path)
          else setParent(path)
          setPicker(null)
        }}
      />
    </div>
  )
}
