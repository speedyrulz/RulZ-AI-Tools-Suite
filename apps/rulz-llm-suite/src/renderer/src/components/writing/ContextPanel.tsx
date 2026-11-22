import { RefreshCw, TriangleAlert, X } from 'lucide-react'
import { useEffect } from 'react'
import type { AssemblyPartKind } from '@shared/novel'
import { formatNumber } from '../../lib/format'
import { ACTION_LABELS, summaryHealth } from '../../lib/novelRun'
import { sceneOf, useNovel } from '../../stores/novel'
import { Badge, Button, Tooltip } from '../ui'
import { PanelSection } from './common'

const PART_LABELS: Record<AssemblyPartKind, string> = {
  system: 'System and style',
  task: 'Task',
  scene: 'This scene',
  prose: 'The text so far',
  fixedEntries: 'Always-on entries',
  triggeredEntries: 'Triggered entries',
  storySoFar: 'Story so far'
}

// Neutral shades, so the bar reads by position and legend; red stays for state.
const PART_SWATCH: Record<AssemblyPartKind, string> = {
  system: 'bg-text/80',
  task: 'bg-text/60',
  scene: 'bg-text/45',
  prose: 'bg-link/70',
  fixedEntries: 'bg-ok/70',
  triggeredEntries: 'bg-ok/45',
  storySoFar: 'bg-warn/60'
}

const WHY: Record<string, string> = { always: 'always on', pov: 'point of view', present: 'present', sticky: 'kept for this scene' }
const reasonText = (r: string): string => WHY[r] ?? r

export function ContextPanel() {
  const target = useNovel((s) => s.target)
  const state = useNovel((s) => s.state)
  const last = useNovel((s) => s.lastRequest)
  const busy = useNovel((s) => s.contextBusy || !!s.gen)
  const hashes = useNovel((s) => s.hashes)
  const sceneId = target?.kind === 'scene' ? target.id : null
  const scene = sceneId ? sceneOf(state, sceneId)?.scene : null
  const { previewContext, removeSticky, summarizeAll } = useNovel.getState()

  // A preview when the tab is shown for a scene that has none yet (the last request, if any, is shown meanwhile).
  useEffect(() => {
    if (sceneId) void previewContext()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sceneId])

  if (!sceneId || !scene || !state) {
    return <div className="p-4 text-xs text-faint">Open a scene to see what a request for it would contain.</div>
  }
  const health = summaryHealth(state.outline, hashes, state.words)
  const built = last?.built
  const result = built?.result
  const used = result ? result.totalTokens : 0
  const limit = result ? result.available + (built?.input.templateOverhead ?? 0) : 0
  const over = !!result && (last?.tooBig || used > limit)
  const stickyEntries = scene.sticky.map((id) => ({ id, entry: state.bible.find((e) => e.id === id) }))

  return (
    <div>
      <PanelSection
        title="Budget"
        action={
          <Button size="sm" variant="ghost" disabled={busy} onClick={() => void previewContext()}>
            <RefreshCw size={13} className={busy ? 'animate-spin' : undefined} /> Refresh
          </Button>
        }
      >
        {!last || !result ? (
          <p className="text-xs text-faint">{busy ? 'Counting…' : 'Nothing counted yet. Choose Refresh, or write something.'}</p>
        ) : (
          <>
            <p className="text-[11px] text-faint">
              {last.source === 'preview' ? `Preview: ${ACTION_LABELS[last.action]} at the cursor.` : `Last request: ${ACTION_LABELS[last.action]}.`}
              {built?.estimated && ' Counts are estimates: load a model for exact ones.'}
            </p>
            <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-track" role="img" aria-label={`${formatNumber(used)} of ${formatNumber(limit)} tokens used`}>
              {result.parts.map((p) => (
                <Tooltip key={p.kind} content={`${PART_LABELS[p.kind]}: ${formatNumber(p.tokens)} tokens`}>
                  <div className={PART_SWATCH[p.kind]} style={{ width: `${Math.min(100, (p.tokens / Math.max(1, limit)) * 100)}%` }} />
                </Tooltip>
              ))}
            </div>
            <div className="flex items-baseline justify-between text-xs tabular-nums">
              <span className={over ? 'text-danger-text' : 'text-body'}>
                {formatNumber(used)} of {formatNumber(limit)} tokens
                {last.actual !== null && <span className="text-faint"> · template count {formatNumber(last.actual)}</span>}
              </span>
              <span className="text-faint">context {formatNumber(last.contextLength)}</span>
            </div>
            <ul className="space-y-1">
              {result.parts.map((p) => (
                <li key={p.kind} className="flex items-center gap-2 text-xs">
                  <span className={`size-2.5 shrink-0 rounded-sm ${PART_SWATCH[p.kind]}`} aria-hidden />
                  <span className="flex-1 text-body">{PART_LABELS[p.kind]}</span>
                  <span className="text-muted tabular-nums">{formatNumber(p.tokens)}</span>
                  <span className="w-20 text-right text-faint tabular-nums">{p.cap !== null ? `of ${formatNumber(p.cap)}` : ''}</span>
                </li>
              ))}
            </ul>
            {over && (
              <p className="flex items-start gap-1.5 text-xs text-danger-text" role="alert">
                <TriangleAlert size={14} className="mt-px shrink-0" /> This request does not fit, so nothing is sent. Shorten the style, beats or entries, lower the reply size in the Project tab, or load a model with a longer context.
              </p>
            )}
          </>
        )}
        {(health.missing.length > 0 || health.stale.length > 0) && (
          <div className="flex items-center gap-2 rounded-lg border border-border bg-panel-2 px-3 py-2 text-xs">
            <span className="flex-1 text-body">
              {health.missing.length > 0 && `${health.missing.length} ${health.missing.length === 1 ? 'summary' : 'summaries'} missing`}
              {health.missing.length > 0 && health.stale.length > 0 && ', '}
              {health.stale.length > 0 && `${health.stale.length} out of date`}
            </span>
            <Button size="sm" disabled={busy || !!useNovel.getState().proposal} onClick={() => void summarizeAll()}>
              Summarize all
            </Button>
          </div>
        )}
      </PanelSection>

      {result && built && (
        <>
          {result.warnings.length > 0 && (
            <PanelSection title="Warnings">
              <ul className="space-y-1.5">
                {result.warnings.map((w) => (
                  <li key={w} className="flex items-start gap-1.5 text-xs text-warn-text">
                    <TriangleAlert size={13} className="mt-px shrink-0" /> {w}
                  </li>
                ))}
              </ul>
            </PanelSection>
          )}
          <PanelSection title="Included entries">
            {result.includedEntries.length === 0 ? (
              <p className="text-xs text-faint">No bible entries in this request. Entries come in when their name, alias or trigger word appears in the beats or nearby text.</p>
            ) : (
              <ul className="space-y-1.5">
                {result.includedEntries.map((id) => {
                  const c = built.candidates[id]
                  const cut = result.dropped.find((d) => d.how === 'cut' && d.label === c?.name)
                  return (
                    <li key={id} className="flex items-center gap-2 text-[13px]">
                      <span className="min-w-0 flex-1 truncate">{c?.name ?? id}</span>
                      {cut && <Badge tone="warn">Cut to first paragraph</Badge>}
                      <Badge>{reasonText(c?.reason ?? '')}</Badge>
                    </li>
                  )
                })}
              </ul>
            )}
          </PanelSection>
          {result.dropped.length > 0 && (
            <PanelSection title="Left out or cut">
              <ul className="space-y-1.5">
                {result.dropped.map((d, i) => (
                  <li key={i} className="flex items-center gap-2 text-xs">
                    <span className="min-w-0 flex-1 truncate text-body">{d.label}</span>
                    <Badge tone={d.how === 'dropped' ? 'warn' : 'neutral'}>{d.how === 'dropped' ? 'Left out' : 'Cut'}</Badge>
                    <span className="text-faint">{d.reason}</span>
                  </li>
                ))}
              </ul>
            </PanelSection>
          )}
          {built.fallbacks.missing + built.fallbacks.stale > 0 && (
            <PanelSection title="Story so far">
              <p className="text-xs text-muted">
                {built.fallbacks.missing > 0 && `${built.fallbacks.missing} earlier ${built.fallbacks.missing === 1 ? 'part has' : 'parts have'} no summary and use the plan instead. `}
                {built.fallbacks.stale > 0 && `${built.fallbacks.stale} ${built.fallbacks.stale === 1 ? 'summary is' : 'summaries are'} out of date and use the plan instead.`}
              </p>
            </PanelSection>
          )}
        </>
      )}

      <PanelSection title="Kept for this scene">
        <p className="text-[11px] text-faint">Entries pulled in by a trigger stay in this scene’s requests until you remove them.</p>
        {stickyEntries.length === 0 ? (
          <p className="text-xs text-faint">None yet.</p>
        ) : (
          <ul className="space-y-1">
            {stickyEntries.map(({ id, entry }) => (
              <li key={id} className="flex items-center gap-2 text-[13px]">
                <span className="min-w-0 flex-1 truncate">{entry?.name ?? `${id} (no such entry)`}</span>
                <Button size="sm" variant="ghost" aria-label={`Remove ${entry?.name ?? id} from this scene`} onClick={() => void removeSticky(scene.id, id)}>
                  <X size={14} /> Remove
                </Button>
              </li>
            ))}
          </ul>
        )}
      </PanelSection>
    </div>
  )
}
