import { AlertTriangle, ChevronDown, FolderOpen, FolderSearch, Plus, Trash2 } from 'lucide-react'
import { useState } from 'react'
import type { BibleKind } from '@shared/novel'
import { invoke } from '../../lib/api'
import { cn } from '../../lib/format'
import { useNovel } from '../../stores/novel'
import { Button, DropdownMenu, IconButton, Select } from '../ui'
import { ConfirmDialog, KIND_LABELS, KindIcon, NameDialog } from './common'
import { TrashDialog } from './HistoryDialog'
import { OutlineTree } from './OutlineTree'

export async function pickProjectFolder(): Promise<void> {
  const root = useNovel.getState().root
  const dir = await invoke('dialog:pickFolder', { title: 'Open a project folder', defaultPath: root || undefined })
  if (dir) await useNovel.getState().openProject(dir)
}

function ProjectBar() {
  const title = useNovel((s) => s.state?.project.title ?? '')
  const folder = useNovel((s) => s.folder)
  const projects = useNovel((s) => s.projects)
  const [creating, setCreating] = useState(false)
  const [trashOpen, setTrashOpen] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const store = useNovel.getState()
  return (
    <div className="flex items-center gap-1.5 border-b border-border p-3">
      <DropdownMenu
        align="start"
        trigger={
          <button
            className="flex h-9 min-w-0 flex-1 items-center gap-2 rounded-lg border border-border bg-panel-2 px-3 text-left text-[13px] font-medium hover:bg-hover"
            aria-label="Project menu"
            onClick={() => void store.refreshProjects()}
          >
            <span className="min-w-0 flex-1 truncate">{title || 'No project'}</span>
            <ChevronDown size={14} className="shrink-0 text-faint" />
          </button>
        }
        items={[
          ...projects.slice(0, 8).map((p) => ({ label: p.folder === folder ? `${p.title} (open)` : p.title, disabled: p.folder === folder, onSelect: () => void store.openProject(p.folder) })),
          ...(projects.length ? (['sep'] as const) : []),
          { label: 'New project…', icon: <Plus size={13} />, onSelect: () => setCreating(true) },
          { label: 'Open folder…', icon: <FolderSearch size={13} />, onSelect: () => void pickProjectFolder() },
          'sep',
          { label: 'Trash…', icon: <Trash2 size={13} />, onSelect: () => setTrashOpen(true) },
          { label: 'Show in folder', icon: <FolderOpen size={13} />, onSelect: () => folder && void invoke('shell:showItem', folder) },
          { label: 'Close project', onSelect: () => void store.closeProject() },
          'sep',
          { label: 'Delete project…', icon: <Trash2 size={13} />, danger: true, onSelect: () => setDeleting(true) }
        ]}
      />
      <NameDialog open={creating} onOpenChange={setCreating} title="New project" label="Title of the novel" confirm="Create" onSubmit={(t) => void store.createProject(t)} />
      <TrashDialog open={trashOpen} onOpenChange={setTrashOpen} />
      <ConfirmDialog
        open={deleting}
        onOpenChange={setDeleting}
        title="Delete this project?"
        body={
          <>
            <p>
              The whole folder <span className="font-mono text-xs break-all">{folder}</span> moves to the Recycle Bin: the manuscript, the bible, the history and the trash.
            </p>
            <p>If it cannot be moved, nothing is deleted.</p>
          </>
        }
        confirm="Move to Recycle Bin"
        onConfirm={() => void store.deleteProject()}
      />
    </div>
  )
}

function BibleList() {
  const bible = useNovel((s) => s.state?.bible ?? [])
  const blocked = useNovel((s) => !!s.state?.blocked)
  const target = useNovel((s) => s.target)
  const [kind, setKind] = useState<BibleKind | 'all'>('all')
  const [creating, setCreating] = useState(false)
  const [newKind, setNewKind] = useState<BibleKind>('character')
  const [deleting, setDeleting] = useState<string | null>(null)
  const store = useNovel.getState()
  const shown = bible.filter((e) => kind === 'all' || e.kind === kind).sort((a, b) => a.name.localeCompare(b.name))
  const del = bible.find((e) => e.id === deleting)
  return (
    <div className="border-t border-border">
      <div className="flex items-center gap-2 px-3 pt-3 pb-1">
        <h2 className="flex-1 text-[11px] font-semibold tracking-wide text-faint uppercase">Story bible</h2>
        <Select className="h-7 w-28 text-xs" aria-label="Filter by kind" value={kind} onChange={(e) => setKind(e.target.value as BibleKind | 'all')}>
          <option value="all">All kinds</option>
          {(Object.keys(KIND_LABELS) as BibleKind[]).map((k) => (
            <option key={k} value={k}>
              {KIND_LABELS[k]}
            </option>
          ))}
        </Select>
        <IconButton
          size="sm"
          label="New entry"
          disabled={blocked}
          onClick={() => {
            setNewKind(kind === 'all' ? 'character' : kind)
            setCreating(true)
          }}
        >
          <Plus size={14} />
        </IconButton>
      </div>
      <div className="space-y-px px-2 pb-3">
        {shown.length === 0 && <div className="px-2 py-3 text-xs text-faint">{bible.length ? 'No entries of this kind.' : 'No entries yet. Add characters, places and lore the model should know.'}</div>}
        {shown.map((e) => (
          <div key={e.id} className={cn('group flex items-center rounded-md', target?.kind === 'entry' && target.id === e.id ? 'bg-hover text-text' : 'text-body hover:bg-hover')}>
            <button className="flex min-w-0 flex-1 items-center gap-2 px-2 py-1.5 text-left text-[13px]" onClick={() => void store.selectEntry(e.id)}>
              <KindIcon kind={e.kind} />
              <span className="min-w-0 flex-1 truncate">{e.name}</span>
              {e.always && <span className="shrink-0 text-[11px] text-faint">always</span>}
              {e.problems.length > 0 && (
                <span className="flex shrink-0 items-center gap-1 text-[11px] text-warn-text" title={e.problems.join('\n')}>
                  <AlertTriangle size={12} /> Problem
                </span>
              )}
            </button>
            <button
              className="rounded p-1 text-faint opacity-0 group-hover:opacity-100 hover:bg-hover hover:text-danger focus-visible:opacity-100"
              aria-label={`Delete ${e.name}`}
              disabled={blocked}
              onClick={() => setDeleting(e.id)}
            >
              <Trash2 size={13} />
            </button>
          </div>
        ))}
      </div>
      <NewEntryDialog open={creating} onOpenChange={setCreating} kind={newKind} onSubmit={(name, k) => void store.createEntry(name, k)} />
      <ConfirmDialog
        open={!!del}
        onOpenChange={(o) => !o && setDeleting(null)}
        title="Delete this entry?"
        body={
          <p>
            <strong>{del?.name}</strong> moves to the project trash. Scenes that name it as POV or present keep the reference until you change it.
          </p>
        }
        confirm="Delete"
        onConfirm={() => del && void store.deleteEntry(del.id)}
      />
    </div>
  )
}

function NewEntryDialog({ open, onOpenChange, kind, onSubmit }: { open: boolean; onOpenChange: (o: boolean) => void; kind: BibleKind; onSubmit: (name: string, kind: BibleKind) => void }) {
  const [k, setK] = useState(kind)
  // The kind chosen in the filter is the starting point each time the dialog opens.
  const [seen, setSeen] = useState(open)
  if (seen !== open) {
    setSeen(open)
    if (open) setK(kind)
  }
  return (
    <NameDialog
      open={open}
      onOpenChange={onOpenChange}
      title="New bible entry"
      label="Name"
      confirm="Create"
      onSubmit={(name) => onSubmit(name, k)}
      extra={
        <Select aria-label="Kind" value={k} onChange={(e) => setK(e.target.value as BibleKind)}>
          {(Object.keys(KIND_LABELS) as BibleKind[]).map((key) => (
            <option key={key} value={key}>
              {KIND_LABELS[key]}
            </option>
          ))}
        </Select>
      }
    />
  )
}

export function LeftColumn() {
  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-border bg-panel">
      <ProjectBar />
      <div className="min-h-0 flex-1 overflow-y-auto">
        <OutlineTree />
        <BibleList />
      </div>
    </aside>
  )
}

/** Shown when no project is open: recent projects, New project and Open folder. */
export function Welcome() {
  const projects = useNovel((s) => s.projects)
  const root = useNovel((s) => s.root)
  const opening = useNovel((s) => s.opening)
  const openError = useNovel((s) => s.openError)
  const [creating, setCreating] = useState(false)
  const store = useNovel.getState()
  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-2xl space-y-6 px-6 py-10">
        <div>
          <h1 className="font-[family-name:var(--font-display)] text-lg font-semibold">Writing</h1>
          <p className="mt-1 text-[13px] text-muted">
            Write a novel with a local model. A project is a folder of plain Markdown files: the manuscript, the story bible and the outline. You can open the same files in other apps.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="primary" onClick={() => setCreating(true)} disabled={opening}>
            <Plus size={15} /> New project
          </Button>
          <Button onClick={() => void pickProjectFolder()} disabled={opening}>
            <FolderSearch size={15} /> Open folder…
          </Button>
          {root && (
            <Button variant="ghost" onClick={() => void invoke('shell:openPath', root)}>
              <FolderOpen size={15} /> Show projects folder
            </Button>
          )}
        </div>
        {openError && (
          <div role="alert" className="flex items-start gap-2 rounded-lg border border-danger-border bg-danger-bg px-3 py-2 text-[13px] text-danger-text">
            <AlertTriangle size={15} className="mt-0.5 shrink-0" /> {openError}
          </div>
        )}
        <section>
          <h2 className="mb-2 text-[11px] font-semibold tracking-wide text-faint uppercase">Your projects</h2>
          {projects.length === 0 ? (
            <p className="text-[13px] text-faint">{root ? `No projects in ${root} yet.` : 'No projects yet.'}</p>
          ) : (
            <div className="space-y-1.5">
              {projects.map((p) => (
                <button key={p.folder} onClick={() => void store.openProject(p.folder)} disabled={opening} className="flex w-full items-center gap-3 rounded-lg border border-border bg-panel px-4 py-3 text-left hover:border-border-strong hover:bg-hover disabled:opacity-50">
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-medium">{p.title}</div>
                    <div className="truncate font-mono text-[11px] text-faint">{p.folder}</div>
                  </div>
                  <div className="shrink-0 text-right text-xs text-muted tabular-nums">
                    <div>{p.words.toLocaleString('en-US')} words</div>
                    <div className="text-faint">{new Date(p.updatedAt).toLocaleDateString()}</div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </section>
      </div>
      <NameDialog open={creating} onOpenChange={setCreating} title="New project" label="Title of the novel" confirm="Create" onSubmit={(t) => void store.createProject(t)} />
    </div>
  )
}
