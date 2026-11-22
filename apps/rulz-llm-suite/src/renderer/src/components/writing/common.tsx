import { CircleCheck, CircleDashed, CircleDot, Circle, MapPin, ScrollText, StickyNote, UserRound } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import type { BibleKind, SceneStatus } from '@shared/novel'
import { cn } from '../../lib/format'
import { Button, Dialog, Field, Input, Textarea } from '../ui'

export const STATUS_LABELS: Record<SceneStatus, string> = { idea: 'Idea', beats: 'Beats', draft: 'Draft', revised: 'Revised' }

/** Icon for a scene's status; always shown with the word (title and label text), never colour alone. */
export function StatusIcon({ status, size = 14, className }: { status: SceneStatus; size?: number; className?: string }) {
  const props = { size, className: cn(status === 'revised' ? 'text-ok' : 'text-faint', className), 'aria-hidden': true }
  if (status === 'idea') return <CircleDashed {...props} />
  if (status === 'beats') return <Circle {...props} />
  if (status === 'draft') return <CircleDot {...props} />
  return <CircleCheck {...props} />
}

export const KIND_LABELS: Record<BibleKind, string> = { character: 'Character', place: 'Place', lore: 'Lore', other: 'Other' }

export function KindIcon({ kind, size = 14, className }: { kind: BibleKind; size?: number; className?: string }) {
  const props = { size, className: cn('text-faint', className), 'aria-hidden': true }
  if (kind === 'character') return <UserRound {...props} />
  if (kind === 'place') return <MapPin {...props} />
  if (kind === 'lore') return <ScrollText {...props} />
  return <StickyNote {...props} />
}

export function PanelSection({ title, action, children }: { title: string; action?: ReactNode; children: ReactNode }) {
  return (
    <section className="space-y-3 border-b border-border px-4 py-4 last:border-b-0">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-[11px] font-semibold tracking-wide text-faint uppercase">{title}</h3>
        {action}
      </div>
      {children}
    </section>
  )
}

/** A dialog asking for a short name. */
export function NameDialog({
  open,
  onOpenChange,
  title,
  label,
  initial = '',
  confirm,
  extra,
  onSubmit
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  label: string
  initial?: string
  confirm: string
  /** A second field shown under the name (its value is kept by the caller). */
  extra?: ReactNode
  onSubmit: (value: string) => void
}) {
  const [value, setValue] = useState(initial)
  useEffect(() => {
    if (open) setValue(initial)
  }, [open, initial])
  const submit = (): void => {
    if (!value.trim()) return
    onSubmit(value.trim())
    onOpenChange(false)
  }
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={title}
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button variant="primary" disabled={!value.trim()} onClick={submit}>
            {confirm}
          </Button>
        </>
      }
    >
      <div className="space-y-3 p-5">
        <Field label={label}>
          <Input autoFocus value={value} onChange={(e) => setValue(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && submit()} />
        </Field>
        {extra && <Field label="Kind">{extra}</Field>}
      </div>
    </Dialog>
  )
}

/** A dialog that asks before something is deleted or discarded. */
export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  body,
  confirm,
  onConfirm,
  danger = true
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  body: ReactNode
  confirm: string
  onConfirm: () => void
  danger?: boolean
}) {
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={title}
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            variant={danger ? 'danger' : 'primary'}
            onClick={() => {
              onOpenChange(false)
              onConfirm()
            }}
          >
            {confirm}
          </Button>
        </>
      }
    >
      <div className="space-y-2 p-5 text-[13px] text-body">{body}</div>
    </Dialog>
  )
}

/**
 * A value edited as text and committed after a pause, on blur and on unmount, so nothing typed is lost
 * when the user clicks elsewhere. Outside changes show up when the field is not being typed in.
 */
export function useDraft(value: string, commit: (v: string) => void, delay = 700): { draft: string; setDraft: (v: string) => void; flush: () => void } {
  const [draft, setDraftState] = useState(value)
  const state = useRef({ draft: value, dirty: false, commit, timer: null as ReturnType<typeof setTimeout> | null })
  state.current.commit = commit
  const flush = (): void => {
    const s = state.current
    if (s.timer) clearTimeout(s.timer)
    s.timer = null
    if (!s.dirty) return
    s.dirty = false
    s.commit(s.draft)
  }
  useEffect(() => {
    if (!state.current.dirty) {
      state.current.draft = value
      setDraftState(value)
    }
  }, [value])
  useEffect(() => () => flush(), [])
  const setDraft = (v: string): void => {
    const s = state.current
    s.draft = v
    s.dirty = true
    setDraftState(v)
    if (s.timer) clearTimeout(s.timer)
    s.timer = setTimeout(flush, delay)
  }
  return { draft, setDraft, flush }
}

export function DraftInput({ value, onCommit, placeholder, className, id }: { value: string; onCommit: (v: string) => void; placeholder?: string; className?: string; id?: string }) {
  const { draft, setDraft, flush } = useDraft(value, onCommit)
  return (
    <Input
      id={id}
      className={className}
      value={draft}
      placeholder={placeholder}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={flush}
      onKeyDown={(e) => e.key === 'Enter' && flush()}
    />
  )
}

export function DraftArea({ value, onCommit, placeholder, rows = 4, className }: { value: string; onCommit: (v: string) => void; placeholder?: string; rows?: number; className?: string }) {
  const { draft, setDraft, flush } = useDraft(value, onCommit)
  return <Textarea rows={rows} className={className} value={draft} placeholder={placeholder} onChange={(e) => setDraft(e.target.value)} onBlur={flush} />
}

/** "Saved 10:15" style time of an ISO string, for the journal banner. */
export function timeLabel(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })
}
