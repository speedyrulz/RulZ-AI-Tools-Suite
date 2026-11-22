import { ChevronLeft, ChevronRight, ListTree, Sparkles } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { findScene } from '@shared/novelOutline'
import { wordCount } from '@shared/novelText'
import { cn, formatNumber } from '../lib/format'
import { neighbourScene } from '../lib/outlineOps'
import { actionProblem, findAction, PROSE_ACTIONS, type EditorSnapshot } from '../lib/runs'
import { startProse } from '../stores/actions'
import { useScene } from '../stores/editor'
import { useProject } from '../stores/project'
import { useUi } from '../stores/ui'
import { BufferTextarea, EditorBanners } from './EditorParts'
import { JobsPill } from './JobCards'
import { Button, Field, IconButton, Sheet, Textarea } from './ui'

let lastAlternatives = 1

/** The scene editor with the AI actions under it. */
export function WriteTab() {
  const state = useProject((s) => s.state)
  const sceneId = useUi((s) => s.sceneId)
  const setScene = useUi((s) => s.setScene)
  const setOutlineOpen = useUi((s) => s.setOutlineOpen)
  const buffer = useScene((s) => s.buffer)
  const ta = useRef<HTMLTextAreaElement>(null)
  /** The last selection seen in the text (kept because tapping a button can collapse the live one). */
  const sel = useRef<{ path: string; start: number; end: number } | null>(null)
  const [snap, setSnap] = useState<EditorSnapshot | null>(null)

  const found = state && sceneId ? findScene(state.outline, sceneId) : null
  if (!state) return null
  if (!found) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-8 text-center">
        <p className="text-[15px] text-muted">{state.outline.chapters.length === 0 ? 'This project has no scenes yet.' : 'Choose a scene to write in.'}</p>
        <Button variant="primary" onClick={() => setOutlineOpen(true)}>
          <ListTree size={17} /> Chapters and scenes
        </Button>
      </div>
    )
  }
  const { scene, chapter } = found
  const prev = neighbourScene(state.outline, scene.id, -1)
  const next = neighbourScene(state.outline, scene.id, 1)

  const openActions = (): void => {
    const el = ta.current
    const b = useScene.getState().buffer
    if (!b) return
    const seen = sel.current && sel.current.path === b.path ? sel.current : null
    const clamp = (n: number): number => Math.max(0, Math.min(n, b.text.length))
    const live = el && el.selectionEnd > el.selectionStart ? { start: el.selectionStart, end: el.selectionEnd } : null
    const pick = live ?? (seen && seen.end > seen.start ? seen : null) ?? { start: el?.selectionStart ?? b.text.length, end: el?.selectionEnd ?? b.text.length }
    setSnap({ text: b.text, baseHash: b.baseHash, selectionStart: clamp(pick.start), selectionEnd: clamp(pick.end) })
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <button className="flex items-center gap-2 border-b border-border px-4 py-2 text-left active:bg-hover" onClick={() => setOutlineOpen(true)}>
        <div className="min-w-0 flex-1">
          <div className="truncate text-[16px] font-medium">{scene.title}</div>
          <div className="truncate text-[12.5px] text-faint">
            {chapter.title} · {formatNumber(wordCount(buffer?.text ?? ''))} words
            {scene.targetWords ? ` of about ${formatNumber(scene.targetWords)}` : ''}
          </div>
        </div>
        <ListTree size={18} className="shrink-0 text-muted" />
      </button>
      <EditorBanners store={useScene} />
      <BufferTextarea store={useScene} taRef={ta} onSelection={(start, end) => { const p = useScene.getState().buffer?.path; if (p) sel.current = { path: p, start, end } }} placeholder={scene.beats ? `Beats: ${scene.beats}` : 'Start writing…'} />
      <JobsPill />
      <div className="flex items-center gap-1 border-t border-border bg-panel px-2 py-2">
        <IconButton label="Previous scene" disabled={!prev} onClick={() => prev && setScene(prev.id)}>
          <ChevronLeft size={21} />
        </IconButton>
        <Button variant="primary" className="flex-1" disabled={!buffer || buffer.readOnly} onClick={openActions}>
          <Sparkles size={17} /> Write with AI
        </Button>
        <IconButton label="Next scene" disabled={!next} onClick={() => next && setScene(next.id)}>
          <ChevronRight size={21} />
        </IconButton>
      </div>
      <ActionSheet snap={snap} onClose={() => setSnap(null)} />
    </div>
  )
}

function ActionSheet({ snap, onClose }: { snap: EditorSnapshot | null; onClose: () => void }) {
  const [actionId, setActionId] = useState('continue')
  const [note, setNote] = useState('')
  const [alts, setAlts] = useState(lastAlternatives)
  const [busy, setBusy] = useState(false)
  const def = findAction(actionId)
  const problem = snap ? actionProblem(def, snap, note) : null
  const selected = snap ? Math.max(0, snap.selectionEnd - snap.selectionStart) : 0
  // An action that needs a selection can't stay chosen when there is none (the sheet was opened again).
  useEffect(() => {
    if (snap && selected === 0 && findAction(actionId).needsSelection) setActionId('continue')
  }, [snap, selected, actionId])

  const run = async (): Promise<void> => {
    if (!snap || problem) return
    setBusy(true)
    lastAlternatives = alts
    try {
      // The base hash is read again now: a save may have finished while the sheet was open.
      const b = useScene.getState().buffer
      const fresh = b ? { ...snap, text: b.text, baseHash: b.baseHash } : snap
      if (await startProse(def, fresh, note, alts)) onClose()
    } finally {
      setBusy(false)
    }
  }

  return (
    <Sheet
      open={!!snap}
      onOpenChange={(o) => !o && onClose()}
      title="Write with AI"
      description={selected > 0 ? `${selected} characters selected` : 'Nothing selected: actions that need a passage are off. Select text in the scene first.'}
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" disabled={!!problem || busy} onClick={() => void run()}>
            <Sparkles size={16} /> Start
          </Button>
        </>
      }
    >
      <div className="space-y-4 px-4 pb-4">
        <div className="grid grid-cols-2 gap-2">
          {PROSE_ACTIONS.map((a) => {
            const off = !!snap && a.needsSelection && selected === 0
            return (
              <button
                key={a.id}
                disabled={off}
                onClick={() => setActionId(a.id)}
                className={cn(
                  'rounded-xl border px-3 py-2.5 text-left text-[14px] disabled:opacity-40',
                  a.id === actionId ? 'border-accent bg-accent-soft' : 'border-border bg-panel-2 active:border-accent/50'
                )}
              >
                {a.label}
              </button>
            )
          })}
        </div>
        <p className="text-[13px] text-muted">{def.hint}</p>
        <Field label="Note" hint={def.needsNote ? 'Needed for this one.' : 'Optional. Tell the AI what you want.'}>
          <Textarea rows={3} value={note} onChange={(e) => setNote(e.target.value)} />
        </Field>
        <Field label="Alternatives" hint="How many versions to write. More takes longer.">
          <div className="flex gap-1.5">
            {[1, 2, 3, 4].map((n) => (
              <button key={n} className={cn('h-11 w-12 rounded-lg border text-[15px]', n === alts ? 'border-accent bg-accent-soft' : 'border-border text-muted active:bg-hover')} onClick={() => setAlts(n)}>
                {n}
              </button>
            ))}
          </div>
        </Field>
        {problem && <p className="text-[13.5px] text-warn">{problem}</p>}
      </div>
    </Sheet>
  )
}
