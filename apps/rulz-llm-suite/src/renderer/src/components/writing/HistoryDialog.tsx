import { RotateCcw } from 'lucide-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import type { NovelHistoryItem, NovelTrashItem } from '@shared/novel'
import { errorMessage, invoke } from '../../lib/api'
import { formatBytes } from '../../lib/format'
import { useNovel } from '../../stores/novel'
import { Button, Dialog } from '../ui'
import { ConfirmDialog } from './common'

/** "20261122T101500" as a readable local time; the stamp is shown as is when it doesn't parse. */
export function stampLabel(stamp: string): string {
  const m = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/.exec(stamp)
  if (!m) return stamp
  const d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]), Number(m[4]), Number(m[5]), Number(m[6]))
  return d.toLocaleString([], { dateStyle: 'medium', timeStyle: 'medium' })
}

/** Snapshots of one file with a preview and Restore (a normal edit, saved with a snapshot of the text it replaces). */
export function HistoryDialog({ path, open, onOpenChange }: { path: string; open: boolean; onOpenChange: (open: boolean) => void }) {
  const folder = useNovel((s) => s.folder)
  const [items, setItems] = useState<NovelHistoryItem[] | null>(null)
  const [chosen, setChosen] = useState<NovelHistoryItem | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [confirm, setConfirm] = useState(false)

  useEffect(() => {
    if (!open || !folder) return
    setItems(null)
    setChosen(null)
    setPreview(null)
    invoke('novel:history', folder, path).then(
      (list) => setItems(list),
      (e) => {
        toast.error(errorMessage(e))
        setItems([])
      }
    )
  }, [open, folder, path])

  useEffect(() => {
    if (!chosen || !folder) return
    let live = true
    setPreview(null)
    invoke('novel:readSnapshot', folder, chosen.snapshot).then(
      (t) => live && setPreview(t),
      (e) => live && setPreview(`Could not read this snapshot: ${errorMessage(e)}`)
    )
    return () => {
      live = false
    }
  }, [chosen, folder])

  return (
    <>
      <Dialog
        open={open}
        onOpenChange={onOpenChange}
        className="w-[min(820px,calc(100vw-32px))]"
        title="History"
        description={`Earlier versions of ${path}. Restoring keeps the text it replaces in the history too.`}
        footer={
          <>
            <Button variant="ghost" onClick={() => onOpenChange(false)}>
              Close
            </Button>
            <Button variant="primary" disabled={!chosen || preview === null} onClick={() => setConfirm(true)}>
              <RotateCcw size={14} /> Restore
            </Button>
          </>
        }
      >
        <div className="flex h-[60vh] min-h-0">
          <div className="w-60 shrink-0 overflow-y-auto border-r border-border p-2">
            {items === null && <div className="p-3 text-xs text-faint">Loading…</div>}
            {items?.length === 0 && <div className="p-3 text-xs text-faint">No earlier versions yet. They are saved as you write.</div>}
            {items?.map((it) => (
              <button
                key={it.snapshot}
                onClick={() => setChosen(it)}
                aria-pressed={chosen?.snapshot === it.snapshot}
                className={`block w-full rounded-md px-2.5 py-1.5 text-left text-[13px] hover:bg-hover ${chosen?.snapshot === it.snapshot ? 'bg-hover text-text' : 'text-body'}`}
              >
                <div>{stampLabel(it.stamp)}</div>
                <div className="text-[11px] text-faint">{formatBytes(it.bytes)}</div>
              </button>
            ))}
          </div>
          <div className="min-w-0 flex-1 overflow-auto p-4">
            {!chosen && <div className="text-xs text-faint">Pick a version to preview it.</div>}
            {chosen && preview === null && <div className="text-xs text-faint">Loading…</div>}
            {preview !== null && <pre className="font-[family-name:var(--font-sans)] text-[13px] leading-relaxed whitespace-pre-wrap text-body">{preview}</pre>}
          </div>
        </div>
      </Dialog>
      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        danger={false}
        title="Restore this version?"
        body="Your current text replaces nothing silently: it is saved to the history first, so you can come back to it."
        confirm="Restore"
        onConfirm={() => {
          if (!chosen) return
          void useNovel.getState().restoreSnapshot(path, chosen.snapshot)
          onOpenChange(false)
        }}
      />
    </>
  )
}

/** Deleted scenes, chapters and entries, with Restore and Empty trash. */
export function TrashDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const folder = useNovel((s) => s.folder)
  const [items, setItems] = useState<NovelTrashItem[] | null>(null)
  const [confirm, setConfirm] = useState(false)

  const load = (): void => {
    if (!folder) return
    invoke('novel:trash', folder).then(setItems, (e) => {
      toast.error(errorMessage(e))
      setItems([])
    })
  }
  useEffect(() => {
    if (open) {
      setItems(null)
      load()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, folder])

  return (
    <>
      <Dialog
        open={open}
        onOpenChange={onOpenChange}
        title="Trash"
        description="Things you deleted from this project. Nothing here is gone until you empty the trash."
        footer={
          <>
            <Button variant="danger" disabled={!items?.length} onClick={() => setConfirm(true)}>
              Empty trash
            </Button>
            <Button variant="ghost" onClick={() => onOpenChange(false)}>
              Close
            </Button>
          </>
        }
      >
        <div className="max-h-[55vh] min-h-24 overflow-y-auto p-2">
          {items === null && <div className="p-3 text-xs text-faint">Loading…</div>}
          {items?.length === 0 && <div className="p-3 text-xs text-faint">The trash is empty.</div>}
          {items?.map((it) => (
            <div key={it.id} className="flex items-center gap-3 rounded-md px-3 py-2 hover:bg-hover">
              <div className="min-w-0 flex-1">
                <div className="truncate text-[13px]">{it.label}</div>
                <div className="text-[11px] text-faint">{new Date(it.deletedAt).toLocaleString()}</div>
              </div>
              <Button
                size="sm"
                onClick={async () => {
                  await useNovel.getState().restoreTrash(it.id)
                  load()
                }}
              >
                <RotateCcw size={14} /> Restore
              </Button>
            </div>
          ))}
        </div>
      </Dialog>
      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title="Empty the trash?"
        body="The trash folder of this project goes to the Recycle Bin. You can no longer restore these items from the app."
        confirm="Empty trash"
        onConfirm={async () => {
          if (!folder) return
          try {
            await invoke('novel:emptyTrash', folder)
            toast.success('The project trash is in the Recycle Bin.')
          } catch (e) {
            toast.error(`Nothing was removed: ${errorMessage(e)}`)
          }
          load()
        }}
      />
    </>
  )
}
