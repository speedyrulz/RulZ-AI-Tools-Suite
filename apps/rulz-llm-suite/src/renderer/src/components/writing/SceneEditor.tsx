import { Check, ChevronDown, Feather, History as HistoryIcon, LoaderCircle, MessageSquareQuote, Minimize2, Maximize2, PenLine, Play, RefreshCw, Square, TriangleAlert, X } from 'lucide-react'
import { useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import type { NovelAction, RewriteVariant } from '@shared/novel'
import { wordCount } from '@shared/novelText'
import { STALE_PENDING, VARIANT_LABELS, pendingStale, scenePath } from '../../lib/novelRun'
import { cn } from '../../lib/format'
import { sceneOf, useNovel } from '../../stores/novel'
import { Button, DropdownMenu, IconButton, Input, Select } from '../ui'
import { BufferBanners } from './Banners'
import { CodeEditor, type PendingProps } from './CodeEditor'
import { HistoryDialog } from './HistoryDialog'

const REWRITE_VARIANTS: RewriteVariant[] = ['tighter', 'vivid', 'tone', 'show', 'custom']

export function SceneEditor({ sceneId }: { sceneId: string }) {
  const scene = useNovel((s) => sceneOf(s.state, sceneId)?.scene)
  const blocked = useNovel((s) => !!s.state?.blocked)
  const path = scene ? scenePath(scene) : ''
  const buffer = useNovel((s) => s.buffers[path])
  const missing = useNovel((s) => s.state?.missing.includes(path) ?? false)
  const gen = useNovel((s) => s.gen)
  const pending = useNovel((s) => s.pending)
  const selection = useNovel((s) => s.selection)
  const [historyOpen, setHistoryOpen] = useState(false)

  const mine = !!pending && pending.path === path
  const hasAlt = mine && !!pending.alts[pending.active]
  const pv = useMemo<PendingProps | null>(() => {
    if (!hasAlt || !pending) return null
    return {
      key: `${pending.id}:${pending.active}`,
      original: pending.baseText,
      proposed: pending.alts[pending.active].text,
      canAccept: () => {
        const s = useNovel.getState()
        return !(s.pending && s.buffers[path] && pendingStale(s.pending, s.buffers[path]))
      },
      onRefuse: () => {
        toast.error(STALE_PENDING)
        useNovel.setState({ pendingRefused: true })
      },
      onResolved: (text) => void useNovel.getState().resolvePending(text)
    }
    // The view restarts only when the pending change or the alternative shown changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pending?.id, pending?.active, hasAlt, path])

  if (!scene) return <div className="p-6 text-sm text-faint">This scene is not in the outline any more.</div>
  if (!buffer) return <div className="p-6 text-sm text-faint">Opening…</div>

  const generating = gen?.path === path
  const readOnly = blocked || buffer.readOnly || missing || generating
  return (
    <div className="flex h-full min-h-0 flex-col">
      <BufferBanners path={path} />
      {mine ? <PendingBar path={path} /> : <Toolbar empty={!buffer.text.trim()} hasSelection={selection.from !== selection.to} disabled={blocked || buffer.readOnly || missing || !!gen} onHistory={() => setHistoryOpen(true)} />}
      {generating && <GenStrip />}
      <div className="min-h-0 flex-1">
        <CodeEditor
          path={path}
          text={buffer.text}
          readOnly={readOnly}
          prose
          register
          cursorAtEnd
          pending={pv}
          placeholder={scene.beats.trim() ? 'Write here, or choose Draft scene to write it from the beats.' : 'Write here. Add beats in the Scene tab to plan the scene.'}
          onChange={(t) => useNovel.getState().edit(path, t)}
          onSelection={(s) => useNovel.getState().setSelection(s)}
        />
      </div>
      <HistoryDialog path={path} open={historyOpen} onOpenChange={setHistoryOpen} />
    </div>
  )
}

function Toolbar({ empty, hasSelection, disabled, onHistory }: { empty: boolean; hasSelection: boolean; disabled: boolean; onHistory: () => void }) {
  const [note, setNote] = useState('')
  const [alts, setAlts] = useState(1)
  const noteRef = useRef<HTMLInputElement>(null)
  const run = (action: NovelAction, variant?: RewriteVariant): void => {
    if ((variant === 'tone' || variant === 'custom') && !note.trim()) {
      noteRef.current?.focus()
      return void toast.message(variant === 'tone' ? 'Write the tone you want in the note field, then choose this again.' : 'Write your instruction in the note field, then choose this again.')
    }
    void useNovel.getState().runAction({ action, variant, note, alternatives: alts })
  }
  const selTip = hasSelection ? undefined : 'Select some text first'
  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-border bg-panel px-3 py-2" role="toolbar" aria-label="Writing actions">
      <Button size="sm" variant={empty ? 'primary' : 'secondary'} disabled={disabled || !empty} title={empty ? 'Write the whole scene from its beats' : 'Draft scene works on an empty scene'} onClick={() => run('draftScene')}>
        <Feather size={14} /> Draft scene
      </Button>
      <Button size="sm" disabled={disabled} title="Write on from the cursor" onClick={() => run('continue')}>
        <Play size={14} /> Continue
      </Button>
      <DropdownMenu
        align="start"
        trigger={
          <Button size="sm" disabled={disabled || !hasSelection} title={selTip ?? 'Rewrite the selected text'}>
            <PenLine size={14} /> Rewrite <ChevronDown size={12} />
          </Button>
        }
        items={REWRITE_VARIANTS.map((v) => ({ label: VARIANT_LABELS[v], onSelect: () => run('rewrite', v) }))}
      />
      <Button size="sm" disabled={disabled || !hasSelection} title={selTip ?? 'Make the selected text longer'} onClick={() => run('expand')}>
        <Maximize2 size={14} /> Expand
      </Button>
      <Button size="sm" disabled={disabled || !hasSelection} title={selTip ?? 'Make the selected text shorter'} onClick={() => run('condense')}>
        <Minimize2 size={14} /> Condense
      </Button>
      <Button size="sm" disabled={disabled} title="Write dialogue for the current beat" onClick={() => run('dialogue')}>
        <MessageSquareQuote size={14} /> Dialogue
      </Button>
      <div className="mx-1 hidden h-5 w-px bg-border lg:block" />
      <Input
        ref={noteRef}
        className="h-8 min-w-40 flex-1 basis-48"
        placeholder="Note to the model (optional)"
        aria-label="Note to the model"
        value={note}
        onChange={(e) => setNote(e.target.value)}
      />
      <Select className="h-8 w-36" aria-label="Alternatives" value={alts} onChange={(e) => setAlts(Number(e.target.value))}>
        {[1, 2, 3, 4].map((n) => (
          <option key={n} value={n}>
            {n === 1 ? '1 version' : `${n} alternatives`}
          </option>
        ))}
      </Select>
      <IconButton label="History of this scene" onClick={onHistory}>
        <HistoryIcon size={16} />
      </IconButton>
    </div>
  )
}

function GenStrip() {
  const gen = useNovel((s) => s.gen)
  if (!gen) return null
  const pct = gen.progress !== null ? Math.round(gen.progress * 100) : null
  const status =
    gen.phase === 'preparing'
      ? 'Preparing the request…'
      : gen.phase === 'waiting'
        ? pct !== null
          ? `Reading the prompt… ${pct}%`
          : 'Waiting for the model…'
        : `Writing${gen.of > 1 ? ` version ${gen.alt} of ${gen.of}` : ''}… ${wordCount(gen.text)} words`
  return (
    <div className="flex items-start gap-3 border-b border-border bg-panel-2 px-4 py-2 text-[13px]">
      <LoaderCircle size={16} className="mt-0.5 shrink-0 animate-spin text-muted" aria-hidden />
      <div className="min-w-0 flex-1">
        <div className="text-body" role="status">
          {gen.label}: {status}
        </div>
        {gen.text && <div className="mt-1 line-clamp-2 text-xs text-faint italic">{gen.text.slice(-260)}</div>}
      </div>
      <Button size="sm" onClick={() => useNovel.getState().stop()}>
        <Square size={13} /> Stop
      </Button>
    </div>
  )
}

function PendingBar({ path }: { path: string }) {
  const pending = useNovel((s) => s.pending)
  const refused = useNovel((s) => s.pendingRefused)
  const stale = useNovel((s) => (s.pending && s.buffers[path] ? pendingStale(s.pending, s.buffers[path]) : false))
  const gen = useNovel((s) => s.gen)
  if (!pending) return null
  const { acceptAll, rejectAll, selectAlt, regenerate } = useNovel.getState()
  const writingMore = !!gen && gen.path === path
  return (
    <div className="border-b border-border bg-panel px-3 py-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[13px] font-medium">{pending.label}</span>
        <span className="text-xs text-faint">Review the changes in the text, then accept or reject each one.</span>
        {pending.alts.length > 1 && (
          <div className="flex items-center gap-1" role="tablist" aria-label="Alternatives">
            {pending.alts.map((_, i) => (
              <button
                key={i}
                role="tab"
                aria-selected={pending.active === i}
                onClick={() => selectAlt(i)}
                className={cn('h-7 rounded-md border px-2.5 text-xs', pending.active === i ? 'border-border-strong bg-hover text-text' : 'border-border text-muted hover:bg-hover')}
              >
                Version {i + 1}
              </button>
            ))}
          </div>
        )}
        {writingMore && (
          <span className="flex items-center gap-1 text-xs text-muted">
            <LoaderCircle size={13} className="animate-spin" aria-hidden /> Writing version {gen!.alt} of {gen!.of}…
          </span>
        )}
        <div className="flex-1" />
        {(stale || refused) && (
          <span className="flex items-center gap-1.5 text-[13px] text-warn-text" role="alert">
            <TriangleAlert size={15} /> {STALE_PENDING}
          </span>
        )}
        {(stale || refused) && (
          <Button size="sm" onClick={() => void regenerate()} disabled={writingMore}>
            <RefreshCw size={14} /> Regenerate
          </Button>
        )}
        {writingMore && (
          <Button size="sm" onClick={() => useNovel.getState().stop()}>
            <Square size={13} /> Stop
          </Button>
        )}
        <Button size="sm" onClick={() => void rejectAll()} disabled={writingMore}>
          <X size={14} /> Reject all
        </Button>
        <Button size="sm" variant="primary" onClick={() => void acceptAll()} disabled={writingMore || stale || pending.alts.length === 0}>
          <Check size={14} /> Accept all
        </Button>
      </div>
    </div>
  )
}
