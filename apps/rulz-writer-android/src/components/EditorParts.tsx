import { CloudOff, FileWarning, History } from 'lucide-react'
import { useEffect, useRef, type ReactNode, type RefObject } from 'react'
import { isDirty } from '../lib/autosave'
import { wordDiff } from '../lib/wordDiff'
import type { EditorStore } from '../stores/editor'
import { cn } from '../lib/format'
import { Badge, Button, Sheet, Spinner, Textarea } from './ui'

/** Saved / Saving / Not saved yet, next to the title. */
export function SaveBadge({ store }: { store: EditorStore }) {
  const buffer = store((s) => s.buffer)
  if (!buffer) return null
  if (buffer.readOnly) return <Badge>Read only</Badge>
  if (buffer.conflict) return <Badge tone="danger">Conflict</Badge>
  if (buffer.saving !== null) return <Badge tone="neutral">Saving…</Badge>
  if (buffer.error) return <Badge tone="warn">Not saved</Badge>
  if (isDirty(buffer)) return <Badge tone="neutral">Unsaved</Badge>
  return <Badge tone="ok">Saved</Badge>
}

/** What the owner needs to know about the open file: a failed save, a change on the computer, an old draft. */
export function EditorBanners({ store }: { store: EditorStore }) {
  const buffer = store((s) => s.buffer)
  const offer = store((s) => s.offer)
  const actions = store.getState()
  if (!buffer) return null
  return (
    <div className="space-y-2 px-3 pt-2 empty:hidden">
      {offer && (
        <Banner icon={<History size={16} />} tone="warn" title="You have unsaved text from before">
          <p>
            {offer.fileChanged
              ? 'The file was also changed on the computer since then. If you bring your text back, you will see both versions before anything is replaced.'
              : 'This is what you had written when the app was closed.'}
          </p>
          <div className="mt-2 flex gap-2">
            <Button size="sm" variant="primary" onClick={actions.restoreDraft}>
              Bring it back
            </Button>
            <Button size="sm" onClick={actions.discardDraft}>
              Throw it away
            </Button>
          </div>
        </Banner>
      )}
      {buffer.changedOnDisk !== 'none' && !buffer.conflict && (
        <Banner icon={<FileWarning size={16} />} tone="warn" title={buffer.changedOnDisk === 'deleted' ? 'This file was removed on the computer' : 'This file changed on the computer'}>
          <p>You have edits that aren’t saved yet.</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {buffer.changedOnDisk === 'changed' && (
              <Button size="sm" onClick={() => void actions.saveNow()}>
                Compare
              </Button>
            )}
            <Button size="sm" onClick={() => void actions.loadFromDisk()}>
              {buffer.changedOnDisk === 'changed' ? 'Use the computer’s text' : 'Load it again'}
            </Button>
          </div>
        </Banner>
      )}
      {buffer.error && !buffer.conflict && (
        <Banner icon={<CloudOff size={16} />} tone="warn" title="Your changes aren’t saved yet">
          <p>{buffer.error} They stay here and I try again.</p>
          <div className="mt-2">
            <Button size="sm" onClick={() => void actions.saveNow()}>
              Try now
            </Button>
          </div>
        </Banner>
      )}
    </div>
  )
}

function Banner({ icon, title, tone, children }: { icon: ReactNode; title: string; tone: 'warn'; children: ReactNode }) {
  return (
    <div className={cn('rounded-xl border p-3 text-[13.5px]', tone === 'warn' && 'border-warn/40 bg-warn/10 text-body')}>
      <div className="mb-1 flex items-center gap-1.5 font-medium text-warn">
        {icon} {title}
      </div>
      {children}
    </div>
  )
}

/** Shown when a save found that the file on the computer is not what the phone started from. */
export function ConflictSheet({ store }: { store: EditorStore }) {
  const buffer = store((s) => s.buffer)
  const conflict = buffer?.conflict ?? null
  const keep = store((s) => s.keepMine)
  const useDisk = store((s) => s.useDisk)
  const diff = conflict && buffer ? wordDiff(conflict.diskText, buffer.text) : []
  return (
    <Sheet
      open={!!conflict}
      onOpenChange={() => undefined}
      dismissable={false}
      full
      title="The file changed on the computer"
      description="It was changed there after you opened it. Choose which text to keep."
      footer={
        <>
          <Button onClick={useDisk}>Use the computer’s text</Button>
          <Button variant="primary" onClick={() => void keep()}>
            Keep mine
          </Button>
        </>
      }
    >
      {conflict && buffer && (
        <div className="space-y-4 px-4 pb-4">
          <section>
            <h3 className="mb-1 text-[13px] font-medium tracking-wide text-faint uppercase">Changes from the computer’s text to yours</h3>
            <div className="max-h-64 overflow-y-auto rounded-lg bg-panel-2 p-2.5 text-[14px] leading-relaxed whitespace-pre-wrap text-body">
              {diff.map((p, i) =>
                p.kind === 'same' ? (
                  <span key={i}>{p.text.length > 260 ? `${p.text.slice(0, 110)} … ${p.text.slice(-110)}` : p.text}</span>
                ) : p.kind === 'add' ? (
                  <span key={i} className="rounded-sm bg-ok/20 text-text">
                    {p.text}
                  </span>
                ) : (
                  <span key={i} className="rounded-sm bg-danger/15 text-danger-text line-through decoration-1">
                    {p.text}
                  </span>
                )
              )}
            </div>
            <p className="mt-1 text-[12px] text-faint">Green is only in your text. Red, crossed out, is only on the computer.</p>
          </section>
          <section>
            <h3 className="mb-1 text-[13px] font-medium tracking-wide text-faint uppercase">On the computer</h3>
            <div className="max-h-56 overflow-y-auto rounded-lg border border-border p-2.5 text-[14px] leading-relaxed whitespace-pre-wrap text-body">{conflict.diskText || '(empty)'}</div>
          </section>
          <section>
            <h3 className="mb-1 text-[13px] font-medium tracking-wide text-faint uppercase">Yours</h3>
            <div className="max-h-56 overflow-y-auto rounded-lg border border-border p-2.5 text-[14px] leading-relaxed whitespace-pre-wrap text-body">{buffer.text || '(empty)'}</div>
          </section>
          <p className="text-[13px] text-muted">Keeping yours saves it over the computer’s text. The computer keeps a copy of the text it replaces in the file’s history.</p>
        </div>
      )}
    </Sheet>
  )
}

/** A multi-line text box that follows the buffer. Its element is handed out so AI actions can read the selection. */
export function BufferTextarea({
  store,
  taRef,
  className,
  placeholder,
  onSelection
}: {
  store: EditorStore
  taRef?: RefObject<HTMLTextAreaElement | null>
  className?: string
  placeholder?: string
  /** The selection as it was last seen (a button tap can collapse it before the tap is handled). */
  onSelection?: (start: number, end: number) => void
}) {
  const buffer = store((s) => s.buffer)
  const loading = store((s) => s.loading)
  const pending = store((s) => s.pendingSelection)
  const locked = store((s) => !!s.offer || s.leaving)
  const own = useRef<HTMLTextAreaElement>(null)
  const ref = taRef ?? own

  useEffect(() => {
    const el = ref.current
    if (!el || !pending || !buffer) return
    el.focus()
    el.setSelectionRange(pending.from, pending.to)
    // Bring the selected passage into view: scroll by how far down it is.
    const ratio = buffer.text.length ? pending.from / buffer.text.length : 0
    el.scrollTop = Math.max(0, ratio * el.scrollHeight - el.clientHeight / 3)
    store.getState().clearSelection()
  }, [pending, buffer, store, ref])

  if (!buffer) return <div className="flex flex-1 items-center justify-center text-faint">{loading ? <Spinner /> : null}</div>
  return (
    <Textarea
      ref={ref}
      className={cn('min-h-0 flex-1 resize-none rounded-none border-0 bg-bg px-4 py-3 text-[16px] leading-[1.7] focus:border-0', className)}
      value={buffer.text}
      readOnly={buffer.readOnly || locked}
      placeholder={placeholder}
      spellCheck
      onChange={(e) => store.getState().edit(e.target.value)}
      onSelect={(e) => onSelection?.(e.currentTarget.selectionStart, e.currentTarget.selectionEnd)}
      onPointerUp={(e) => onSelection?.(e.currentTarget.selectionStart, e.currentTarget.selectionEnd)}
      onKeyUp={(e) => onSelection?.(e.currentTarget.selectionStart, e.currentTarget.selectionEnd)}
      onBlur={(e) => onSelection?.(e.currentTarget.selectionStart, e.currentTarget.selectionEnd)}
    />
  )
}
