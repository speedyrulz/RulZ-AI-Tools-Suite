import { ChevronRight, CornerLeftUp, Folder, HardDrive } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { FolderListing } from '@shared/types'
import { errorMessage, invoke } from '../lib/api'
import { Button, Sheet, Spinner } from './ui'

/** Browses folders on the computer, for "Open existing folder" and where new projects go. */
export function FolderPicker({
  open,
  onOpenChange,
  title,
  start,
  action,
  onPick
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  start: string
  action: string
  onPick: (path: string) => void
}) {
  const [listing, setListing] = useState<FolderListing | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  // Only the newest request's answer is used: answers can arrive out of order.
  const latest = useRef(0)
  const go = async (path: string): Promise<void> => {
    const mine = ++latest.current
    setLoading(true)
    setError('')
    try {
      const next = await invoke('fs:listFolders', path)
      if (mine === latest.current) setListing(next)
    } catch (e) {
      if (mine !== latest.current) return
      setError(errorMessage(e))
      if (!listing) {
        const root = await invoke('fs:listFolders', '').catch(() => null)
        if (mine === latest.current) setListing(root)
      }
    } finally {
      if (mine === latest.current) setLoading(false)
    }
  }

  useEffect(() => {
    if (open) void go(start)
  }, [open])

  const sep = listing?.path.endsWith('\\') ? '' : '\\'
  return (
    <Sheet
      open={open}
      onOpenChange={onOpenChange}
      full
      title={title}
      description={listing?.path || 'This computer'}
      footer={
        <Button variant="primary" size="lg" className="w-full" disabled={!listing?.path || loading} onClick={() => listing && onPick(listing.path)}>
          {action}
        </Button>
      }
    >
      <div className="px-2 pb-2">
        {error && <p className="px-2 py-2 text-sm text-danger">{error}</p>}
        {loading && !listing && (
          <p className="flex items-center gap-2 px-2 py-4 text-sm text-faint">
            <Spinner /> Loading…
          </p>
        )}
        {listing && listing.parent !== null && (
          <button className="flex w-full items-center gap-3 rounded-lg px-3 py-3 text-left text-[15px] text-muted active:bg-hover" onClick={() => void go(listing.parent!)}>
            <CornerLeftUp size={18} /> Up one level
          </button>
        )}
        {listing?.folders.map((f) => (
          <button
            key={f}
            className="flex w-full items-center gap-3 rounded-lg px-3 py-3 text-left text-[15px] active:bg-hover"
            onClick={() => void go(listing.path ? `${listing.path}${sep}${f}` : f)}
          >
            {listing.path ? <Folder size={18} className="shrink-0 text-accent" /> : <HardDrive size={18} className="shrink-0 text-accent" />}
            <span className="min-w-0 flex-1 break-all">{f}</span>
            <ChevronRight size={16} className="shrink-0 text-faint" />
          </button>
        ))}
        {listing && listing.path && listing.folders.length === 0 && <p className="px-3 py-3 text-sm text-faint">No folders in here.</p>}
      </div>
    </Sheet>
  )
}
