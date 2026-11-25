import { FileText, RotateCw, X } from 'lucide-react'
import type { PdfAttachment } from '@shared/pdf'
import { formatBytes } from '../lib/format'
import { usePdf } from '../stores/pdf'
import { Button, IconButton, Spinner } from './ui'

/** A PDF in a sent message: name, pages, size. */
export function SentPdfChip({ pdf }: { pdf: PdfAttachment }) {
  return (
    <span className="flex max-w-full min-w-0 items-center gap-2 rounded-xl border border-border bg-panel px-2.5 py-1.5 text-[13px] text-text">
      <FileText size={15} className="shrink-0 text-muted" aria-hidden />
      <span className="min-w-0 truncate">{pdf.name}</span>
      <span className="shrink-0 text-faint tabular-nums">
        {pdf.pages} {pdf.pages === 1 ? 'page' : 'pages'} · {formatBytes(pdf.bytes)}
      </span>
    </span>
  )
}

/** The PDFs waiting to go with the next message, above the text box. */
export function PdfChips({ className }: { className?: string }) {
  const chips = usePdf((s) => s.chips)
  const remove = usePdf((s) => s.remove)
  const retry = usePdf((s) => s.retry)
  if (!chips.length) return null
  return (
    <ul className={className ?? 'mb-2.5 space-y-2'} aria-label="PDFs to send">
      {chips.map((c) => {
        const uploading = c.state === 'uploading' || c.state === 'queued'
        const pct = c.bytes ? Math.round((c.sent / c.bytes) * 100) : 0
        return (
          <li key={c.key} className="flex items-center gap-1 rounded-xl border border-border bg-panel-2 pl-3">
            <FileText size={16} className="shrink-0 text-muted" aria-hidden />
            <div className="min-w-0 flex-1 py-1.5 text-[13px]">
              <div className="flex items-baseline gap-2">
                <span className="min-w-0 truncate">{c.name}</span>
                <span className="shrink-0 text-faint tabular-nums">
                  {c.attachment ? `${c.attachment.pages} ${c.attachment.pages === 1 ? 'page' : 'pages'} · ` : ''}
                  {formatBytes(c.bytes)}
                </span>
              </div>
              {uploading && (
                <div className="mt-1 flex items-center gap-2 text-faint" role="status">
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-track">
                    <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="tabular-nums">{c.state === 'queued' ? 'Waiting' : `${pct}%`}</span>
                  {c.state === 'uploading' && <Spinner className="shrink-0" />}
                </div>
              )}
              {c.state === 'error' && (
                <div className="mt-1 flex flex-wrap items-center gap-x-2" role="alert">
                  <span className="min-w-0 text-danger">{c.error}</span>
                  <Button size="sm" variant="ghost" className="-ml-2" onClick={() => retry(c.key)}>
                    <RotateCw size={14} aria-hidden /> Try again
                  </Button>
                </div>
              )}
              {c.note && <div className="mt-1 text-danger">{c.note}</div>}
            </div>
            <IconButton label={uploading ? `Cancel sending ${c.name}` : `Remove ${c.name}`} onClick={() => void remove(c.key)}>
              <X size={18} />
            </IconButton>
          </li>
        )
      })}
    </ul>
  )
}
