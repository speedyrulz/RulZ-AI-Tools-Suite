import { LoaderCircle, RefreshCw, ScanSearch, Square, TriangleAlert } from 'lucide-react'
import { useMemo } from 'react'
import type { CheckFinding, FindingType } from '@shared/novel'
import { findingKey } from '@shared/novelChecks'
import { hashText, scenePath, visibleFindings, type FindingsEntry } from '../../lib/novelRun'
import type { FindingSpec } from '../../lib/novelEditor'
import { sceneOf, useNovel } from '../../stores/novel'
import { Badge, Button } from '../ui'
import { PanelSection } from './common'

export const FINDING_LABELS: Record<FindingType, string> = {
  continuity: 'Continuity',
  character: 'Character',
  timeline: 'Timeline',
  pov: 'Point of view',
  tense: 'Tense',
  other: 'Other'
}

const FINDING_TONE: Record<FindingType, 'neutral' | 'accent' | 'warn' | 'danger' | 'ok'> = {
  continuity: 'warn',
  character: 'warn',
  timeline: 'accent',
  pov: 'neutral',
  tense: 'neutral',
  other: 'neutral'
}

/** The findings of a scene that are not dismissed, and the same as specs for the editor (stable between renders). */
export function useVisibleFindings(sceneId: string | null): { entry: FindingsEntry | null; visible: CheckFinding[]; specs: FindingSpec[]; dismissedCount: number } {
  const entry = useNovel((s) => (sceneId ? (s.findings[sceneId] ?? null) : null))
  const dismissed = useNovel((s) => (sceneId ? s.state?.dismissed?.[sceneId] : undefined))
  return useMemo(() => {
    const visible = entry ? visibleFindings(entry.items, dismissed) : []
    return {
      entry,
      visible,
      specs: visible.map((f) => ({ id: findingKey(f), type: f.type, quote: f.quote })),
      dismissedCount: entry ? entry.items.length - visible.length : 0
    }
  }, [entry, dismissed])
}

export function ChecksPanel() {
  const target = useNovel((s) => s.target)
  const state = useNovel((s) => s.state)
  const gen = useNovel((s) => s.gen)
  const checkRun = useNovel((s) => s.checkRun)
  const issue = useNovel((s) => s.checkIssue)
  const findings = useNovel((s) => s.findings)
  const sceneId = target?.kind === 'scene' ? target.id : null
  const scene = sceneId ? sceneOf(state, sceneId)?.scene : null
  const path = scene ? scenePath(scene) : ''
  const text = useNovel((s) => s.buffers[path]?.text)
  const { entry, visible, dismissedCount } = useVisibleFindings(sceneId)
  const stale = useMemo(() => !!entry && text !== undefined && hashText(text) !== entry.textHash, [entry, text])
  const store = useNovel.getState()
  if (!state) return null
  if (!sceneId || !scene) return <div className="p-4 text-xs text-faint">Open a scene to check it.</div>

  const busy = !!gen || !!checkRun
  const noDismiss = state.blocked ? 'The project is read-only.' : state.checksProblem ? `Dismissals cannot be saved: ${state.checksProblem}` : null
  const others = Object.entries(findings)
    .filter(([id, e]) => id !== sceneId && e.items.length > 0 && sceneOf(state, id))
    .map(([id, e]) => ({ id, title: sceneOf(state, id)!.scene.title, n: visibleFindings(e.items, state.dismissed?.[id]).length }))
    .filter((o) => o.n > 0)
  const running = gen && (gen.action === 'checkScene' || checkRun)

  return (
    <div>
      <PanelSection
        title="Consistency check"
        action={
          running ? (
            <Button size="sm" onClick={() => store.stop()}>
              <Square size={13} /> Stop
            </Button>
          ) : (
            <Button size="sm" variant="primary" disabled={busy || !text?.trim()} title={text?.trim() ? 'Check this scene against the bible and what is known so far' : 'The scene has no text to check'} onClick={() => void store.checkScene(sceneId)}>
              <ScanSearch size={14} /> Check scene
            </Button>
          )
        }
      >
        {running ? (
          <p className="flex items-center gap-2 text-[13px] text-body" role="status">
            <LoaderCircle size={15} className="animate-spin text-muted" aria-hidden />
            {gen?.label ?? 'Checking…'}
            {checkRun ? ` (${checkRun.index + 1} of ${checkRun.sceneIds.length})` : ''}
          </p>
        ) : (
          <p className="text-[11px] text-faint">The model reads the scene against the bible, the story so far and the tracked state. Nothing in your text changes until you accept a fix.</p>
        )}
        {issue && issue.sceneId === sceneId && (
          <div className="space-y-2 rounded-md border border-danger-border bg-danger-bg/40 p-3">
            <p className="flex items-start gap-1.5 text-[13px] text-danger-text" role="alert">
              <TriangleAlert size={15} className="mt-0.5 shrink-0" /> {issue.error || 'The reply could not be used.'} Nothing was changed.
            </p>
            {issue.raw && (
              <details className="text-xs">
                <summary className="cursor-pointer text-muted">What the model wrote</summary>
                <pre className="mt-1.5 max-h-48 overflow-auto rounded-md border border-border bg-code p-2 font-mono text-[11px] whitespace-pre-wrap text-body">{issue.raw}</pre>
              </details>
            )}
            <div className="flex justify-end gap-2">
              <Button size="sm" variant="ghost" onClick={() => store.clearCheckIssue()}>
                Dismiss
              </Button>
              <Button size="sm" disabled={busy} onClick={() => void store.checkScene(sceneId)}>
                <RefreshCw size={14} /> Retry
              </Button>
            </div>
          </div>
        )}
      </PanelSection>

      <PanelSection title={`Findings${entry ? ` (${visible.length})` : ''}`}>
        {!entry ? (
          <p className="text-xs text-faint">Not checked yet.</p>
        ) : (
          <>
            {stale && (
              <p className="flex items-start gap-1.5 text-xs text-warn-text" role="status">
                <TriangleAlert size={13} className="mt-px shrink-0" /> The text changed since this check. The findings may be out of date; check again.
              </p>
            )}
            {visible.length === 0 && <p className="text-xs text-faint">{entry.items.length === 0 ? 'Nothing found.' : 'Every finding is dismissed.'}</p>}
            <ul className="space-y-3">
              {visible.map((f) => {
                const key = findingKey(f)
                const located = entry.located[key]
                return (
                  <li key={key} className="space-y-1.5 rounded-md border border-border bg-panel-2 p-3 text-[13px]">
                    <div className="flex items-center gap-2">
                      <Badge tone={FINDING_TONE[f.type]}>{FINDING_LABELS[f.type]}</Badge>
                      {!located && <span className="text-[11px] text-faint">Not found in the text</span>}
                    </div>
                    <blockquote className="border-l-2 border-border-strong pl-2 text-xs text-muted italic">{f.quote}</blockquote>
                    <p className="text-body">{f.problem}</p>
                    {f.conflictsWith && (
                      <p className="text-xs text-muted">
                        <span className="text-faint">Conflicts with: </span>
                        {f.conflictsWith}
                      </p>
                    )}
                    {f.suggestion && (
                      <p className="text-xs text-muted">
                        <span className="text-faint">Suggestion: </span>
                        {f.suggestion}
                      </p>
                    )}
                    <div className="flex flex-wrap justify-end gap-1.5 pt-0.5">
                      <Button size="sm" variant="ghost" disabled={!located} title={located ? 'Select this text in the scene' : 'The quote was not found in the text'} onClick={() => store.jumpToFinding(sceneId, f)}>
                        Jump
                      </Button>
                      <Button size="sm" disabled={busy || !located || state.blocked !== null} title={located ? 'Rewrite this passage with the suggestion as the note' : 'The quote was not found in the text'} onClick={() => void store.fixFinding(sceneId, f)}>
                        Fix
                      </Button>
                      <Button size="sm" variant="ghost" disabled={!!noDismiss} title={noDismiss ?? 'Hide this finding for this scene'} onClick={() => void store.dismissFinding(sceneId, f)}>
                        Dismiss
                      </Button>
                    </div>
                  </li>
                )
              })}
            </ul>
          </>
        )}
        {dismissedCount > 0 && (
          <div className="flex items-center gap-2 text-xs text-faint">
            <span className="flex-1">{dismissedCount} dismissed</span>
            <Button size="sm" variant="ghost" disabled={!!noDismiss} onClick={() => void store.restoreDismissed(sceneId)}>
              Bring back
            </Button>
          </div>
        )}
      </PanelSection>

      {others.length > 0 && (
        <PanelSection title="Other scenes">
          <ul className="space-y-1">
            {others.map((o) => (
              <li key={o.id} className="flex items-center gap-2 text-[13px]">
                <span className="min-w-0 flex-1 truncate">{o.title}</span>
                <span className="text-xs text-faint">{o.n}</span>
                <Button size="sm" variant="ghost" onClick={() => void store.selectScene(o.id)}>
                  Open
                </Button>
              </li>
            ))}
          </ul>
        </PanelSection>
      )}
    </div>
  )
}
