import type { ReactNode } from 'react'
import { Button, Dialog } from '../ui'

/**
 * A question before something that costs money or loses data. It says what happens in plain lines, and focus starts
 * on Cancel so an Enter or a stray key never confirms. The confirm button is disabled while the call is running.
 */
export function ConfirmDialog({
  open,
  title,
  lines,
  warnings,
  confirmLabel,
  variant = 'secondary',
  busy,
  onConfirm,
  onCancel,
  children
}: {
  open: boolean
  title: string
  lines: string[]
  warnings?: string[]
  confirmLabel: string
  variant?: 'primary' | 'secondary' | 'danger'
  busy?: boolean
  onConfirm: () => void
  onCancel: () => void
  children?: ReactNode
}) {
  return (
    <Dialog
      open={open}
      onOpenChange={(o) => !o && onCancel()}
      title={title}
      footer={
        <>
          <Button autoFocus variant="ghost" onClick={onCancel}>
            Cancel
          </Button>
          <Button type="button" variant={variant} disabled={busy} onClick={onConfirm}>
            {confirmLabel}
          </Button>
        </>
      }
    >
      <div className="space-y-2.5 px-5 py-4 text-[13px] text-body">
        {lines.map((l, i) => (
          <p key={i}>{l}</p>
        ))}
        {warnings?.map((w, i) => (
          <p key={i} className="rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">
            {w}
          </p>
        ))}
        {children}
      </div>
    </Dialog>
  )
}
