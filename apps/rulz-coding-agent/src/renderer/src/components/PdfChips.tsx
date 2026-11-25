import { FileText, LoaderCircle, X } from 'lucide-react'
import { cn } from '../lib/format'

/** One attached PDF: its name, its pages, a remove button while it is still unsent, and a spinner while it is read. */
export function PdfChip({
  name,
  pages,
  busy,
  onRemove,
  className
}: {
  name: string
  pages?: number
  busy?: boolean
  onRemove?: () => void
  className?: string
}) {
  const detail = busy ? 'Reading…' : pages ? `${pages} ${pages === 1 ? 'page' : 'pages'}` : ''
  return (
    <span
      className={cn(
        'inline-flex h-7 max-w-[260px] min-w-0 items-center gap-1.5 rounded-full border border-border bg-panel-2 pl-2.5 text-[12.5px] text-text',
        onRemove ? 'pr-1' : 'pr-2.5',
        className
      )}
      aria-busy={busy || undefined}
    >
      {busy ? <LoaderCircle size={14} className="shrink-0 animate-spin text-accent" aria-hidden /> : <FileText size={14} className="shrink-0 text-muted" aria-hidden />}
      <span className="min-w-0 truncate" title={name}>
        {name}
      </span>
      {detail && <span className="shrink-0 text-faint">{detail}</span>}
      {onRemove && (
        <button
          type="button"
          aria-label={`Remove ${name}`}
          title={busy ? 'Wait until it has been read' : `Remove ${name}`}
          disabled={busy}
          onClick={onRemove}
          className="inline-flex size-5 shrink-0 items-center justify-center rounded-full text-muted transition-colors hover:bg-hover hover:text-text disabled:pointer-events-none disabled:opacity-40"
        >
          <X size={14} aria-hidden />
        </button>
      )}
    </span>
  )
}
