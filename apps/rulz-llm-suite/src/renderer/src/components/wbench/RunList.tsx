import { AlertTriangle, Trash2 } from 'lucide-react'
import { useState } from 'react'
import type { BenchRunEntry } from '@shared/writingBench'
import { runProgress } from '../../lib/wbenchRunner'
import { entryId, useWbench } from '../../stores/wbench'
import { Button, Dialog, IconButton } from '../ui'
import { StatusBadge } from './RunView'

/** The saved runs, newest first; a damaged file is listed with its problem and can be deleted. */
export function RunList() {
  const entries = useWbench((s) => s.entries)
  const loaded = useWbench((s) => s.loaded)
  const activeId = useWbench((s) => s.activeId)
  const open = useWbench((s) => s.open)
  const remove = useWbench((s) => s.remove)
  const [confirm, setConfirm] = useState<BenchRunEntry | null>(null)

  if (!loaded) return <p className="text-[13px] text-faint">Reading your runs…</p>
  if (entries.length === 0) return <p className="text-[13px] text-faint">No runs yet. Start one to compare models or adapters on the same scenes.</p>

  return (
    <>
      <div className="space-y-1.5">
        {entries.map((e) => {
          const id = entryId(e)
          if (!e.ok) {
            return (
              <div key={id} className="flex items-center gap-3 rounded-lg border border-danger-border bg-danger-bg px-4 py-3">
                <AlertTriangle size={16} className="shrink-0 text-danger-text" aria-hidden />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium text-danger-text">Damaged run file</div>
                  <div className="truncate font-mono text-[11px] text-muted" title={e.file}>
                    {e.file}
                  </div>
                  <div className="text-[12px] text-danger-text">{e.problem}</div>
                  <div className="text-[11.5px] text-muted">The app does not change or delete it by itself.</div>
                </div>
                <Button size="sm" variant="danger" onClick={() => setConfirm(e)}>
                  <Trash2 size={13} aria-hidden /> Delete
                </Button>
              </div>
            )
          }
          const r = e.run
          const p = runProgress(r)
          return (
            <div key={id} className="flex items-center gap-1 rounded-lg border border-border bg-panel hover:border-border-strong hover:bg-hover">
              <button onClick={() => open(id)} className="flex min-w-0 flex-1 items-center gap-3 px-4 py-3 text-left">
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium">{r.label || 'Untitled run'}</div>
                  <div className="truncate text-[11.5px] text-faint">
                    {new Date(r.createdAt).toLocaleString()} · {r.candidates.length} candidates · {r.prompts.length} prompts · {r.seeds.length} seeds
                  </div>
                </div>
                <div className="shrink-0 text-right text-xs text-muted tabular-nums">
                  <div>
                    {p.finished} / {p.total} outputs
                  </div>
                  <div className="text-faint">
                    {p.voted} / {p.pairs} votes
                  </div>
                </div>
                <StatusBadge status={r.status} />
              </button>
              <IconButton label={activeId === id ? 'Stop the run before deleting it' : 'Delete this run'} disabled={activeId === id} onClick={() => setConfirm(e)} className="mr-1">
                <Trash2 size={15} aria-hidden />
              </IconButton>
            </div>
          )
        })}
      </div>
      <Dialog
        open={confirm !== null}
        onOpenChange={(o) => !o && setConfirm(null)}
        title="Delete this run?"
        description="Its outputs and votes are deleted from disk. This cannot be undone."
        footer={
          <>
            <Button size="sm" onClick={() => setConfirm(null)}>
              Keep it
            </Button>
            <Button
              size="sm"
              variant="danger"
              onClick={() => {
                if (confirm) void remove(entryId(confirm))
                setConfirm(null)
              }}
            >
              Delete
            </Button>
          </>
        }
      >
        <div className="px-5 py-4 text-[13px] text-muted">{confirm?.ok ? (confirm.run.label || 'Untitled run') : (confirm?.file ?? '')}</div>
      </Dialog>
    </>
  )
}
