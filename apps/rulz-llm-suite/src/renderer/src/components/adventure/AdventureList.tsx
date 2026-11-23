import { MoreHorizontal, Pencil, Plus, Signpost, Trash2 } from 'lucide-react'
import { useState } from 'react'
import type { AdventureMeta } from '@shared/adventure'
import { relativeTime } from '../../lib/format'
import { useAdventure } from '../../stores/adventure'
import { Badge, Button, DropdownMenu } from '../ui'
import { ConfirmDialog, NameDialog } from '../writing/common'
import { NewAdventureDialog } from './NewAdventureDialog'
import { FOCUS } from './parts'

export function AdventureList() {
  const list = useAdventure((s) => s.list)
  const loaded = useAdventure((s) => s.listLoaded)
  const opening = useAdventure((s) => s.opening)
  const [creating, setCreating] = useState(false)
  const [renaming, setRenaming] = useState<AdventureMeta | null>(null)
  const [deleting, setDeleting] = useState<AdventureMeta | null>(null)
  const store = useAdventure.getState()

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-2xl space-y-6 px-6 py-10">
        <div>
          <h1 className="font-[family-name:var(--font-display)] text-lg font-semibold">Adventure</h1>
          <p className="mt-1 text-[13px] text-muted">
            Play a choose-your-own-adventure story with your local model. It writes a passage and offers choices, or you type what you do. Every path is kept, so you can go back and try another.
          </p>
        </div>
        <div>
          <Button variant="primary" onClick={() => setCreating(true)}>
            <Plus size={15} aria-hidden /> New adventure
          </Button>
        </div>
        <section>
          <h2 className="mb-2 text-[11px] font-semibold tracking-wide text-faint uppercase">Your adventures</h2>
          {!loaded ? (
            <p className="text-[13px] text-faint">Loading…</p>
          ) : list.length === 0 ? (
            <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-border py-10 text-center text-[13px] text-faint">
              <Signpost size={26} aria-hidden />
              No adventures yet. Start one with New adventure.
            </div>
          ) : (
            <div className="space-y-1.5">
              {list.map((m) => (
                <div key={m.id} className="flex items-center gap-3 rounded-lg border border-border bg-panel px-4 py-3 hover:border-border-strong">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-sm font-medium">{m.title}</span>
                      {m.ended && <Badge>Ended</Badge>}
                    </div>
                    <div className="truncate text-[12px] text-muted">
                      {[m.genre, m.worldTitle ? `From ${m.worldTitle}` : ''].filter(Boolean).join(' · ') || 'No genre'}
                    </div>
                  </div>
                  <div className="shrink-0 text-right text-xs text-muted tabular-nums">
                    <div>
                      {m.nodeCount} {m.nodeCount === 1 ? 'passage' : 'passages'}
                    </div>
                    <div className="text-faint">Played {relativeTime(m.updatedAt)}</div>
                  </div>
                  <Button size="sm" disabled={opening} onClick={() => void store.open(m.id)}>
                    Open
                  </Button>
                  <DropdownMenu
                    trigger={
                      <button aria-label={`More for ${m.title}`} className={`rounded-md p-1.5 text-muted hover:bg-hover hover:text-text ${FOCUS}`}>
                        <MoreHorizontal size={16} />
                      </button>
                    }
                    items={[
                      { label: 'Rename', icon: <Pencil size={14} />, onSelect: () => setRenaming(m) },
                      'sep',
                      { label: 'Delete', icon: <Trash2 size={14} />, danger: true, onSelect: () => setDeleting(m) }
                    ]}
                  />
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
      <NewAdventureDialog open={creating} onOpenChange={setCreating} />
      <NameDialog
        open={!!renaming}
        onOpenChange={(v) => !v && setRenaming(null)}
        title="Rename adventure"
        label="Title"
        initial={renaming?.title ?? ''}
        confirm="Rename"
        onSubmit={(t) => renaming && void store.rename(renaming.id, t)}
      />
      <ConfirmDialog
        open={!!deleting}
        onOpenChange={(v) => !v && setDeleting(null)}
        title="Delete this adventure?"
        body={<p>“{deleting?.title}” and all its paths go to the Recycle Bin.</p>}
        confirm="Delete"
        onConfirm={() => deleting && void store.remove(deleting.id)}
      />
    </div>
  )
}
