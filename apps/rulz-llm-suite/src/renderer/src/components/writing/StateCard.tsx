import { Check, LoaderCircle, RefreshCw, Square, TriangleAlert, X } from 'lucide-react'
import { wordCount } from '@shared/novelText'
import { cn } from '../../lib/format'
import { useNovel } from '../../stores/novel'
import { Button } from '../ui'

/** The result of Update state: each change is accepted or rejected, and nothing is written until Apply. */
export function StateCard() {
  const p = useNovel((s) => s.stateProposal)
  const gen = useNovel((s) => s.gen)
  if (!p) return null
  const store = useNovel.getState()
  const accepted = p.items.filter((i) => i.accepted).length

  return (
    <section className="space-y-3 border-b border-border bg-panel-2 px-4 py-4" aria-label={p.title}>
      <div>
        <h3 className="text-[11px] font-semibold tracking-wide text-faint uppercase">Proposal</h3>
        <div className="mt-0.5 text-[13px] font-medium">{p.title}</div>
      </div>

      {p.status === 'running' && (
        <>
          <div className="flex items-center gap-2 text-[13px] text-body" role="status">
            <LoaderCircle size={15} className="animate-spin text-muted" aria-hidden />
            {gen?.phase === 'waiting' && gen.progress !== null ? `Reading the prompt… ${Math.round(gen.progress * 100)}%` : gen?.text ? `Writing… ${wordCount(gen.text)} words` : 'Preparing…'}
          </div>
          <div className="flex justify-end">
            <Button size="sm" onClick={() => store.discardStateProposal()}>
              <Square size={13} /> Stop
            </Button>
          </div>
        </>
      )}

      {p.status === 'invalid' && (
        <>
          <p className="flex items-start gap-1.5 text-[13px] text-danger-text" role="alert">
            <TriangleAlert size={15} className="mt-0.5 shrink-0" /> {p.error ?? 'The reply could not be used.'} Nothing was changed.
          </p>
          {p.raw && (
            <details className="text-xs">
              <summary className="cursor-pointer text-muted">What the model wrote</summary>
              <pre className="mt-1.5 max-h-48 overflow-auto rounded-md border border-border bg-code p-2 font-mono text-[11px] whitespace-pre-wrap text-body">{p.raw}</pre>
            </details>
          )}
          <div className="flex justify-end gap-2">
            <Button size="sm" variant="ghost" onClick={() => store.discardStateProposal()}>
              Dismiss
            </Button>
            <Button
              size="sm"
              onClick={() => {
                store.discardStateProposal()
                void store.updateState(p.sceneId)
              }}
            >
              <RefreshCw size={14} /> Retry
            </Button>
          </div>
        </>
      )}

      {(p.status === 'ready' || p.status === 'applied') && (
        <>
          {p.status === 'ready' && p.items.length === 0 && <p className="text-[13px] text-body">No changes to the state were found in this scene.</p>}
          {p.status === 'ready' && (
            <ul className="max-h-80 space-y-2 overflow-y-auto">
              {p.items.map((i) => (
                <li key={i.key} className={cn('flex items-start gap-2 rounded-md border bg-panel p-2 text-[13px]', i.accepted ? 'border-border' : 'border-border opacity-60')}>
                  <span className={cn('min-w-0 flex-1', !i.accepted && 'line-through')}>{i.label}</span>
                  <Button size="sm" variant={i.accepted ? 'secondary' : 'ghost'} aria-pressed={i.accepted} aria-label={`Accept: ${i.label}`} onClick={() => store.setStateItem(i.key, true)}>
                    <Check size={13} /> Accept
                  </Button>
                  <Button size="sm" variant={i.accepted ? 'ghost' : 'secondary'} aria-pressed={!i.accepted} aria-label={`Reject: ${i.label}`} onClick={() => store.setStateItem(i.key, false)}>
                    <X size={13} /> Reject
                  </Button>
                </li>
              ))}
            </ul>
          )}
          {p.left.length > 0 && (
            <div className="text-[11px] text-warn-text">
              Left out:
              <ul className="mt-0.5 list-disc pl-4">
                {p.left.map((l, i) => (
                  <li key={i}>{l}</li>
                ))}
              </ul>
            </div>
          )}
          {p.status === 'applied' && (
            <p className="text-[13px] text-body" role="status">
              {p.applied} {p.applied === 1 ? 'change' : 'changes'} applied.
            </p>
          )}
          {p.rejected.length > 0 && (
            <div className="text-xs text-danger-text" role="alert">
              Not applied:
              <ul className="mt-0.5 list-disc pl-4">
                {p.rejected.map((r, i) => (
                  <li key={i}>
                    {r.label}: {r.reason}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {p.refused && (
            <p className="flex items-start gap-1.5 text-[13px] text-warn-text" role="alert">
              <TriangleAlert size={15} className="mt-0.5 shrink-0" /> {p.refused}
            </p>
          )}
          <div className="flex justify-end gap-2">
            <Button size="sm" variant="ghost" onClick={() => store.discardStateProposal()}>
              {p.status === 'applied' ? 'Close' : 'Reject all'}
            </Button>
            {p.status === 'ready' && p.refused && (
              <Button
                size="sm"
                onClick={() => {
                  store.discardStateProposal()
                  void store.updateState(p.sceneId)
                }}
              >
                <RefreshCw size={14} /> Regenerate
              </Button>
            )}
            {p.status === 'ready' && (
              <Button size="sm" variant="primary" disabled={accepted === 0 || p.applying || !!p.refused} onClick={() => void store.applyStateProposal()}>
                <Check size={14} /> Apply accepted{accepted > 0 ? ` (${accepted})` : ''}
              </Button>
            )}
          </div>
        </>
      )}
    </section>
  )
}
