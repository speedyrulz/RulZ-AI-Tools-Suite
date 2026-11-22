import { MessagesSquare, Plus, Trash2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import type { BibleEntry, BibleKind } from '@shared/novel'
import { entryPath } from '../lib/outlineOps'
import { errorMessage } from '../lib/format'
import { link } from '../lib/remote'
import { discardDraftFor, useEntry } from '../stores/editor'
import { useProject } from '../stores/project'
import { useUi } from '../stores/ui'
import { BufferTextarea, ConflictSheet, EditorBanners, SaveBadge } from './EditorParts'
import { Badge, Button, Field, Input, Select, Sheet } from './ui'

const KINDS: { value: BibleKind; label: string; plural: string }[] = [
  { value: 'character', label: 'Character', plural: 'Characters' },
  { value: 'place', label: 'Place', plural: 'Places' },
  { value: 'lore', label: 'Lore', plural: 'Lore' },
  { value: 'other', label: 'Other', plural: 'Other' }
]

/** The story bible: entries by kind, opened as their own file (front matter and all). */
export function BibleTab() {
  const state = useProject((s) => s.state)
  const [openId, setOpenId] = useState<string | null>(null)
  const [adding, setAdding] = useState(false)
  if (!state) return null

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-lg space-y-5 px-4 py-4">
        <Button className="w-full" onClick={() => setAdding(true)}>
          <Plus size={17} /> New entry
        </Button>
        {state.bible.length === 0 && <p className="py-8 text-center text-[14px] text-faint">The story bible is empty. Add characters, places and notes the AI should know about.</p>}
        {KINDS.map((k) => {
          const entries = state.bible.filter((e) => e.kind === k.value)
          if (!entries.length) return null
          return (
            <section key={k.value} className="rounded-2xl border border-border bg-panel p-1.5">
              <h2 className="px-3 pt-2 pb-1 text-[13px] font-medium tracking-wide text-faint uppercase">{k.plural}</h2>
              {entries.map((e) => (
                <button key={e.id} className="flex w-full items-center gap-2 rounded-xl px-3 py-2.5 text-left active:bg-hover" onClick={() => setOpenId(e.id)}>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[15px] font-medium">{e.name}</span>
                    {e.aliases.length > 0 && <span className="block truncate text-[12.5px] text-faint">Also: {e.aliases.join(', ')}</span>}
                  </span>
                  {e.always && <Badge tone="accent">Always</Badge>}
                  {e.problems.length > 0 && <Badge tone="warn">Check</Badge>}
                </button>
              ))}
            </section>
          )
        })}
      </div>
      <EntrySheet id={openId} onClose={() => setOpenId(null)} />
      <NewEntrySheet open={adding} onOpenChange={setAdding} onCreated={(e) => setOpenId(e.id)} />
    </div>
  )
}

function NewEntrySheet({ open, onOpenChange, onCreated }: { open: boolean; onOpenChange: (o: boolean) => void; onCreated: (e: BibleEntry) => void }) {
  const [name, setName] = useState('')
  const [kind, setKind] = useState<BibleKind>('character')
  const [busy, setBusy] = useState(false)
  const create = async (): Promise<void> => {
    const folder = useProject.getState().state?.folder
    if (!folder || !name.trim()) return
    setBusy(true)
    try {
      const entry = await link.invoke('novel:createEntry', folder, name.trim(), kind)
      setName('')
      onOpenChange(false)
      await useProject.getState().reopen()
      onCreated(entry)
    } catch (e) {
      toast.error(errorMessage(e))
    } finally {
      setBusy(false)
    }
  }
  return (
    <Sheet
      open={open}
      onOpenChange={onOpenChange}
      title="New entry"
      footer={
        <>
          <Button onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button variant="primary" disabled={!name.trim() || busy} onClick={() => void create()}>
            Create
          </Button>
        </>
      }
    >
      <div className="space-y-4 px-4 pb-4">
        <Field label="Name">
          <Input autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder="Mara Venn" />
        </Field>
        <Field label="Kind">
          <Select value={kind} onChange={(e) => setKind(e.target.value as BibleKind)}>
            {KINDS.map((k) => (
              <option key={k.value} value={k.value}>
                {k.label}
              </option>
            ))}
          </Select>
        </Field>
      </div>
    </Sheet>
  )
}

/** One entry file. The front matter (name, aliases, triggers…) is part of the text and is saved as it is. */
function EntrySheet({ id, onClose }: { id: string | null; onClose: () => void }) {
  const state = useProject((s) => s.state)
  const entry = state?.bible.find((e) => e.id === id)
  const buffer = useEntry((s) => s.buffer)
  const [confirm, setConfirm] = useState(false)
  const setCharacter = useUi((s) => s.setCharacter)
  const setTab = useUi((s) => s.setTab)

  useEffect(() => {
    if (id && state) void useEntry.getState().open(state.folder, entryPath(id))
    // Opens when the id changes; the project folder is stable while an entry is open.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id])

  const close = async (): Promise<boolean> => {
    // Stays open while a conflict waits for an answer: its sheet is shown instead.
    if (!(await useEntry.getState().leave())) return false
    onClose()
    // The list shows names and flags from the file, which may have changed.
    void useProject.getState().reopen()
    return true
  }

  return (
    <>
      <Sheet
        full
        open={!!id}
        onOpenChange={(o) => !o && void close()}
        title={entry?.name ?? id ?? ''}
        description={entry ? KINDS.find((k) => k.value === entry.kind)?.label : undefined}
      >
        <div className="flex h-full min-h-[40dvh] flex-col">
          <div className="flex items-center gap-2 px-4 pb-2">
            <SaveBadge store={useEntry} />
            <div className="flex-1" />
            {entry?.kind === 'character' && (
              <Button
                size="sm"
                onClick={async () => {
                  if (!(await close())) return
                  setCharacter(entry.id)
                  setTab('interview')
                }}
              >
                <MessagesSquare size={15} /> Interview
              </Button>
            )}
            <Button size="sm" variant="danger" onClick={() => setConfirm(true)}>
              <Trash2 size={15} /> Delete
            </Button>
          </div>
          <EditorBanners store={useEntry} />
          {entry?.problems.length ? <p className="px-4 pt-2 text-[12.5px] text-warn">{entry.problems.join(' ')}</p> : null}
          <BufferTextarea store={useEntry} className="min-h-[30dvh]" />
          {buffer && <p className="px-4 py-2 text-[12px] text-faint">The lines between the dashes at the top set the name, aliases and when the AI sees this entry. Change them with care.</p>}
        </div>
      </Sheet>
      <ConflictSheet store={useEntry} />
      <Sheet
        open={confirm}
        onOpenChange={setConfirm}
        title={`Delete “${entry?.name ?? ''}”?`}
        footer={
          <>
            <Button onClick={() => setConfirm(false)}>Cancel</Button>
            <Button
              variant="danger"
              onClick={async () => {
                setConfirm(false)
                if (!entry) return
                // The file is going away: no save, and its draft is forgotten.
                useEntry.getState().drop()
                await discardDraftFor(state!.folder, entryPath(entry.id))
                if (await useProject.getState().deleteEntry(entry.id)) {
                  onClose()
                  toast.success('Deleted.')
                }
              }}
            >
              Delete
            </Button>
          </>
        }
      >
        <p className="px-4 pb-4 text-[15px] text-muted">This removes the entry from the story bible on the computer.</p>
      </Sheet>
    </>
  )
}
