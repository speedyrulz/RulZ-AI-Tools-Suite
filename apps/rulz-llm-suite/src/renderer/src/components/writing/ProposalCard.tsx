import { Check, LoaderCircle, Pencil, RefreshCw, Square, TriangleAlert, X } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { wordCount } from '@shared/novelText'
import { resolveSceneRefs } from '../../lib/novelRun'
import { dropPlanItem, isPlanList, planEmptied } from '../../lib/proposalPlan'
import { useNovel, type Proposal } from '../../stores/novel'
import { Button, Input, Textarea } from '../ui'

/** An AI result that is not text in the manuscript: nothing is written until you accept it. */
export function ProposalCard() {
  const proposal = useNovel((s) => s.proposal)
  const gen = useNovel((s) => s.gen)
  const queue = useNovel((s) => s.summarizeQueue.length)
  const bible = useNovel((s) => s.state?.bible ?? [])
  const [draft, setDraft] = useState<Proposal | null>(null)
  const [editing, setEditing] = useState(false)

  useEffect(() => {
    setDraft(proposal ? structuredClone(proposal) : null)
    setEditing(false)
    // A new proposal, or the same one becoming ready, starts a fresh draft.
  }, [proposal?.id, proposal?.status])

  if (!proposal) return null
  const { acceptProposal, rejectProposal, retryProposal, stop } = useNovel.getState()

  if (proposal.status === 'running') {
    return (
      <Card title={proposal.title}>
        <div className="flex items-center gap-2 text-[13px] text-body" role="status">
          <LoaderCircle size={15} className="animate-spin text-muted" aria-hidden />
          {gen?.phase === 'waiting' && gen.progress !== null ? `Reading the prompt… ${Math.round(gen.progress * 100)}%` : gen?.text ? `Writing… ${wordCount(gen.text)} words` : 'Preparing…'}
        </div>
        <div className="flex justify-end">
          <Button size="sm" onClick={() => stop()}>
            <Square size={13} /> Stop
          </Button>
        </div>
      </Card>
    )
  }

  if (proposal.status === 'invalid') {
    return (
      <Card title={proposal.title} tone="danger">
        <p className="flex items-start gap-1.5 text-[13px] text-danger-text" role="alert">
          <TriangleAlert size={15} className="mt-0.5 shrink-0" /> {proposal.error ?? 'The reply could not be used.'} Nothing was changed.
        </p>
        {proposal.raw && (
          <details className="text-xs">
            <summary className="cursor-pointer text-muted">What the model wrote</summary>
            <pre className="mt-1.5 max-h-48 overflow-auto rounded-md border border-border bg-code p-2 font-mono text-[11px] whitespace-pre-wrap text-body">{proposal.raw}</pre>
          </details>
        )}
        <div className="flex justify-end gap-2">
          <Button size="sm" variant="ghost" onClick={() => rejectProposal()}>
            Dismiss
          </Button>
          <Button size="sm" onClick={() => void retryProposal()}>
            <RefreshCw size={14} /> Retry
          </Button>
        </div>
      </Card>
    )
  }

  const d = draft ?? proposal
  const set = (patch: Partial<Proposal>): void => setDraft({ ...d, ...patch })
  const emptied = planEmptied(d)
  const drop = (i: number): void => setDraft(dropPlanItem(d, i))

  return (
    <Card title={proposal.title} note={queue > 0 ? `${queue} more to summarize after this one.` : undefined}>
      {d.summary && (editing ? (
        <Textarea rows={7} aria-label="Summary" value={d.summary.summary} onChange={(e) => set({ summary: { summary: e.target.value } })} />
      ) : (
        <p className="max-h-60 overflow-y-auto text-[13px] leading-relaxed whitespace-pre-wrap text-body">{d.summary.summary}</p>
      ))}
      {d.chapters && (
        <ol className="max-h-80 space-y-2 overflow-y-auto">
          {d.chapters.chapters.map((c, i) => (
            <li key={i} className="rounded-md border border-border bg-panel p-2 text-[13px]">
              {editing ? (
                <div className="space-y-1.5">
                  <div className="flex gap-1.5">
                    <Input aria-label="Chapter title" value={c.title} onChange={(e) => set({ chapters: { chapters: d.chapters!.chapters.map((x, j) => (j === i ? { ...x, title: e.target.value } : x)) } })} />
                    <Input className="w-28" aria-label="Act" placeholder="Act" value={c.act} onChange={(e) => set({ chapters: { chapters: d.chapters!.chapters.map((x, j) => (j === i ? { ...x, act: e.target.value } : x)) } })} />
                    <Button size="sm" variant="ghost" aria-label="Remove this chapter" onClick={() => set({ chapters: { chapters: d.chapters!.chapters.filter((_, j) => j !== i) } })}>
                      <X size={14} />
                    </Button>
                  </div>
                  <Textarea rows={3} aria-label="Chapter summary" value={c.summary} onChange={(e) => set({ chapters: { chapters: d.chapters!.chapters.map((x, j) => (j === i ? { ...x, summary: e.target.value } : x)) } })} />
                </div>
              ) : (
                <div className="flex items-start gap-1.5">
                  <div className="min-w-0 flex-1">
                    <div className="font-medium">
                      {i + 1}. {c.title} {c.act && <span className="text-[11px] font-normal text-faint">{c.act}</span>}
                    </div>
                    <div className="mt-0.5 text-xs text-muted">{c.summary}</div>
                  </div>
                  <DropItem label={`Leave out chapter ${i + 1}`} onClick={() => drop(i)} />
                </div>
              )}
            </li>
          ))}
        </ol>
      )}
      {d.scenes && (() => {
        const miss = resolveSceneRefs(bible, d.scenes.scenes).unresolved
        return miss.length > 0 ? <p className="text-[11px] text-warn-text">Not in the bible: {miss.join(', ')}. They are left out of POV and present.</p> : null
      })()}
      {d.scenes && (
        <ol className="max-h-80 space-y-2 overflow-y-auto">
          {d.scenes.scenes.map((sc, i) => {
            const name = (id: string): string => bible.find((e) => e.id === id)?.name ?? id
            const shown = resolveSceneRefs(bible, [sc]).scenes[0]
            return (
              <li key={i} className="rounded-md border border-border bg-panel p-2 text-[13px]">
                {editing ? (
                  <div className="space-y-1.5">
                    <div className="flex gap-1.5">
                      <Input aria-label="Scene title" value={sc.title} onChange={(e) => set({ scenes: { scenes: d.scenes!.scenes.map((x, j) => (j === i ? { ...x, title: e.target.value } : x)) } })} />
                      <Button size="sm" variant="ghost" aria-label="Remove this scene" onClick={() => set({ scenes: { scenes: d.scenes!.scenes.filter((_, j) => j !== i) } })}>
                        <X size={14} />
                      </Button>
                    </div>
                    <Textarea rows={4} aria-label="Beats" value={sc.beats} onChange={(e) => set({ scenes: { scenes: d.scenes!.scenes.map((x, j) => (j === i ? { ...x, beats: e.target.value } : x)) } })} />
                  </div>
                ) : (
                  <div className="flex items-start gap-1.5">
                    <div className="min-w-0 flex-1">
                      <div className="font-medium">
                        {i + 1}. {sc.title}
                      </div>
                      <div className="mt-0.5 text-xs whitespace-pre-wrap text-muted">{sc.beats}</div>
                      {(shown.pov || shown.present.length > 0) && (
                        <div className="mt-1 text-[11px] text-faint">
                          {shown.pov && <>POV: {name(shown.pov)}. </>}
                          {shown.present.length > 0 && <>Present: {shown.present.map(name).join(', ')}.</>}
                        </div>
                      )}
                    </div>
                    <DropItem label={`Leave out scene ${i + 1}`} onClick={() => drop(i)} />
                  </div>
                )}
              </li>
            )
          })}
        </ol>
      )}
      {emptied && <p className="text-[11px] text-muted">Every item is left out. Reject the proposal, or press Retry for a new one.</p>}
      <div className="flex justify-end gap-2">
        <Button size="sm" variant="ghost" onClick={() => rejectProposal()} title={isPlanList(d) ? 'Throws away the whole list. Use the X on an item to leave out just that one.' : undefined}>
          <X size={14} /> {isPlanList(d) ? 'Reject all' : 'Reject'}
        </Button>
        {emptied && (
          <Button size="sm" onClick={() => void retryProposal()}>
            <RefreshCw size={14} /> Retry
          </Button>
        )}
        <Button size="sm" onClick={() => setEditing((e) => !e)}>
          <Pencil size={14} /> {editing ? 'Done editing' : 'Edit'}
        </Button>
        <Button size="sm" variant="primary" disabled={emptied} onClick={() => void acceptProposal(d)}>
          <Check size={14} /> Accept
        </Button>
      </div>
    </Card>
  )
}

/** Leaves one planned chapter or scene out of the proposal; the rest can still be accepted. */
function DropItem({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <Button size="sm" variant="ghost" aria-label={label} title={label} onClick={onClick}>
      <X size={14} />
    </Button>
  )
}

function Card({ title, note, tone, children }: { title: string; note?: string; tone?: 'danger'; children: ReactNode }) {
  return (
    <section className={`space-y-3 border-b px-4 py-4 ${tone === 'danger' ? 'border-danger-border bg-danger-bg/40' : 'border-border bg-panel-2'}`} aria-label={title}>
      <div>
        <h3 className="text-[11px] font-semibold tracking-wide text-faint uppercase">Proposal</h3>
        <div className="mt-0.5 text-[13px] font-medium">{title}</div>
        {note && <div className="mt-0.5 text-[11px] text-faint">{note}</div>}
      </div>
      {children}
    </section>
  )
}
