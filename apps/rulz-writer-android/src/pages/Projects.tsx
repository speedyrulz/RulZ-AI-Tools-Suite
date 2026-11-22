import { BookOpen, Plus, RefreshCw, Settings as SettingsIcon } from 'lucide-react'
import { useEffect, useState } from 'react'
import { LogoMark } from '../components/Logo'
import { Button, IconButton, Input, Sheet, Spinner, TopBar } from '../components/ui'
import { formatNumber, relativeTime } from '../lib/format'
import { useLink } from '../stores/link'
import { useProject } from '../stores/project'
import { useUi } from '../stores/ui'

/** The projects in the computer's Writing folder: open one, or start a new one. */
export function Projects() {
  const projects = useProject((s) => s.projects)
  const loading = useProject((s) => s.listLoading)
  const error = useProject((s) => s.listError)
  const loadList = useProject((s) => s.loadList)
  const go = useUi((s) => s.go)
  const computer = useLink((s) => s.computers.find((c) => c.id === s.currentId))
  const [creating, setCreating] = useState(false)
  const [title, setTitle] = useState('')
  const [opening, setOpening] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => void loadList(), [loadList])

  const open = async (folder: string): Promise<void> => {
    setOpening(folder)
    try {
      if (await useProject.getState().open(folder)) go('workspace')
    } finally {
      setOpening(null)
    }
  }

  const create = async (): Promise<void> => {
    if (!title.trim()) return
    setBusy(true)
    try {
      if (await useProject.getState().create(title)) {
        setCreating(false)
        setTitle('')
        go('workspace')
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex h-full flex-col">
      <TopBar>
        <div className="flex size-11 items-center justify-center">
          <LogoMark className="h-6" />
        </div>
        <div className="min-w-0 flex-1 px-1">
          <h1 className="font-[family-name:var(--font-display)] text-[17px] font-semibold">Projects</h1>
          <p className="truncate text-[12px] text-faint">On {computer?.name ?? 'your computer'}</p>
        </div>
        <IconButton label="Refresh" onClick={() => void loadList()} disabled={loading}>
          {loading ? <Spinner /> : <RefreshCw size={19} />}
        </IconButton>
        <IconButton label="Settings" onClick={() => go('settings')}>
          <SettingsIcon size={20} />
        </IconButton>
      </TopBar>
      <div className="px-safe pb-safe min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-lg space-y-3 px-4 py-4">
          <Button size="lg" variant="primary" className="w-full" onClick={() => setCreating(true)}>
            <Plus size={18} /> New project
          </Button>
          {error && <p className="rounded-lg border border-danger-border bg-danger-bg p-3 text-[14px] text-danger-text">{error}</p>}
          {!loading && !error && projects.length === 0 && <p className="py-10 text-center text-[14px] text-faint">No projects yet. Start one here, or in the Writing page on the computer.</p>}
          {projects.map((p) => (
            <button key={p.folder} className="flex w-full items-center gap-3 rounded-2xl border border-border bg-panel px-4 py-3.5 text-left active:bg-hover disabled:opacity-60" disabled={!!opening} onClick={() => void open(p.folder)}>
              <BookOpen size={21} className="shrink-0 text-accent" />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[16px] font-medium">{p.title}</span>
                <span className="block truncate text-[13px] text-faint">
                  {formatNumber(p.words)} words · {relativeTime(Date.parse(p.updatedAt) || 0)}
                </span>
              </span>
              {opening === p.folder && <Spinner className="text-accent" />}
            </button>
          ))}
        </div>
      </div>
      <Sheet
        open={creating}
        onOpenChange={setCreating}
        title="New project"
        description="It is made in the Writing folder on the computer."
        footer={
          <>
            <Button onClick={() => setCreating(false)}>Cancel</Button>
            <Button variant="primary" disabled={!title.trim() || busy} onClick={() => void create()}>
              Create
            </Button>
          </>
        }
      >
        <div className="px-4 pb-4">
          <Input autoFocus value={title} onChange={(e) => setTitle(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && void create()} placeholder="Title of the story" />
        </div>
      </Sheet>
    </div>
  )
}
