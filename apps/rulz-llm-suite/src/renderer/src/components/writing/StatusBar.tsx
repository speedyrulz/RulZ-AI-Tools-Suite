import { CircleAlert, CircleCheck, LoaderCircle, TriangleAlert } from 'lucide-react'
import type { ReactNode } from 'react'
import { wordCount } from '@shared/novelText'
import { formatNumber } from '../../lib/format'
import { scenePath } from '../../lib/novelRun'
import { entryPath, liveWords, sceneOf, useNovel } from '../../stores/novel'

export function StatusBar() {
  const target = useNovel((s) => s.target)
  const state = useNovel((s) => s.state)
  const buffers = useNovel((s) => s.buffers)
  const last = useNovel((s) => s.lastRequest)
  const found = target?.kind === 'scene' ? sceneOf(state, target.id) : null
  const path = target?.kind === 'scene' ? (found ? scenePath(found.scene) : null) : target?.kind === 'entry' ? entryPath(target.id) : null
  const b = path ? buffers[path] : undefined
  const snap = { buffers, state }
  const scene = found && path ? wordCount(b?.text ?? '') : 0
  const chapter = found ? (state?.outline.chapters.find((c) => c.id === found.chapterId)?.scenes.reduce((n, s) => n + liveWords(snap, scenePath(s)), 0) ?? 0) : 0
  const book = state ? state.outline.chapters.reduce((n, c) => n + c.scenes.reduce((m, s) => m + liveWords(snap, scenePath(s)), 0), 0) : 0

  let save: { icon: ReactNode; text: string; tone: string } | null = null
  if (b) {
    if (b.conflict) save = { icon: <CircleAlert size={13} />, text: 'Conflict', tone: 'text-danger-text' }
    else if (b.error) save = { icon: <CircleAlert size={13} />, text: 'Save failed', tone: 'text-danger-text' }
    else if (b.changedOnDisk) save = { icon: <TriangleAlert size={13} />, text: 'Changed on disk', tone: 'text-warn-text' }
    else if (b.saving || b.dirty) save = { icon: <LoaderCircle size={13} className="animate-spin" />, text: 'Saving…', tone: 'text-muted' }
    else save = { icon: <CircleCheck size={13} />, text: 'Saved', tone: 'text-muted' }
  }
  return (
    <div className="flex h-8 shrink-0 items-center gap-4 border-t border-border bg-panel px-4 text-xs text-muted tabular-nums" role="status">
      {found && <span>Scene {formatNumber(scene)} words</span>}
      {found && <span>Chapter {formatNumber(chapter)}</span>}
      <span>Book {formatNumber(book)}</span>
      <div className="flex-1" />
      {last && (
        <span title="Tokens of the last request, of the model's context">
          Request {formatNumber(last.actual ?? last.built.result.totalTokens)} / {formatNumber(last.contextLength)} tokens
          {last.built.estimated && ' (estimated)'}
        </span>
      )}
      {save && (
        <span className={`flex items-center gap-1 ${save.tone}`}>
          {save.icon} {save.text}
        </span>
      )}
    </div>
  )
}
