import { Check, Plus, RefreshCw, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'
import type { BibleEntry, StateFact, StateKind } from '@shared/novel'
import type { WritingJob, WritingStateItem } from '@shared/phone'
import { findScene } from '@shared/novelOutline'
import { link } from '../lib/remote'
import { errorMessage } from '../lib/format'
import { isRunning } from '../lib/jobs'
import { addFactOps, endFactOps, entryName, factLine, factsForScene, kindLabel, otherEntry, removeFactOps, STATE_KINDS, validateFact } from '../lib/stateOps'
import { startUpdateState } from '../stores/actions'
import { useJobs, useProjectJobs } from '../stores/jobs'
import { useProject } from '../stores/project'
import { useUi } from '../stores/ui'
import { JobShell } from './JobCards'
import { Button, Field, Input, Select, Sheet, Spinner } from './ui'

/** Tracked state for the scene being written: what is true there, Update state proposals, manual edits. */
export function StateTab() {
  const state = useProject((s) => s.state)
  const sceneId = useUi((s) => s.sceneId)
  const jobs = useProjectJobs(state?.folder).filter((j) => j.request.action === 'updateState')
  const [adding, setAdding] = useState(false)
  const [busy, setBusy] = useState(false)
  if (!state) return null
  const found = sceneId ? findScene(state.outline, sceneId) : null
  if (!found) return <p className="px-6 py-10 text-center text-[14px] text-faint">Choose a scene on the Write page first.</p>
  const facts = factsForScene(state, found.scene.id)
  const running = jobs.some(isRunning)
  const problem = state.state.problem

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-lg space-y-4 px-4 py-4">
        <p className="text-[13.5px] text-muted">What is true in the story at “{found.scene.title}”. The AI sees these facts when it writes later scenes.</p>
        {problem && <p className="rounded-lg border border-danger-border bg-danger-bg p-3 text-[14px] text-danger-text">The tracked state can’t be changed: {problem}</p>}
        <div className="flex flex-wrap gap-2">
          <Button
            variant="primary"
            disabled={running || busy || !!problem}
            onClick={async () => {
              setBusy(true)
              try {
                await startUpdateState()
              } finally {
                setBusy(false)
              }
            }}
          >
            {running || busy ? <Spinner /> : <RefreshCw size={16} />} Update state
          </Button>
          <Button disabled={!!problem} onClick={() => setAdding(true)}>
            <Plus size={16} /> Add a fact
          </Button>
        </div>

        {jobs.map((job) => (job.status === 'done' && job.result?.kind === 'state' ? <StateProposal key={job.id} job={job} /> : <JobShell key={job.id} job={job} />))}

        <FactSection title="Recorded in this scene" empty="Nothing recorded from this scene yet." facts={facts.added} sceneId={found.scene.id} canEnd={false} />
        <FactSection title="Ended in this scene" empty="Nothing ended here." facts={facts.ended} sceneId={found.scene.id} canEnd={false} />
        <FactSection title="True after this scene" empty="Nothing is tracked yet." facts={facts.after} sceneId={found.scene.id} canEnd />
      </div>
      <AddFactSheet open={adding} onOpenChange={setAdding} sceneId={found.scene.id} />
    </div>
  )
}

function FactSection({ title, empty, facts, sceneId, canEnd }: { title: string; empty: string; facts: StateFact[]; sceneId: string; canEnd: boolean }) {
  const bible = useProject((s) => s.state?.bible ?? [])
  const statePatch = useProject((s) => s.statePatch)
  return (
    <section className="rounded-2xl border border-border bg-panel p-3">
      <h2 className="mb-1.5 text-[13px] font-medium tracking-wide text-faint uppercase">{title}</h2>
      {facts.length === 0 && <p className="text-[13.5px] text-faint">{empty}</p>}
      <ul className="space-y-2">
        {facts.map((f) => (
          <li key={f.id} className="flex items-start gap-2 text-[14.5px]">
            <span className="min-w-0 flex-1">{factLine(f, bible)}</span>
            {canEnd && (
              <Button size="sm" variant="ghost" onClick={() => void statePatch(endFactOps(f.id, sceneId))}>
                End here
              </Button>
            )}
            <Button size="sm" variant="ghost" aria-label="Remove this fact" className="text-danger" onClick={() => void statePatch(removeFactOps(f.id))}>
              <Trash2 size={15} />
            </Button>
          </li>
        ))}
      </ul>
    </section>
  )
}

function itemText(item: WritingStateItem, bible: BibleEntry[]): string {
  if (item.op === 'add' && item.fact) {
    const who = entryName(bible, item.fact.entity)
    const kind = item.fact.kind === 'relationship' && item.fact.about ? `relationship with ${entryName(bible, item.fact.about)}` : item.fact.kind
    return `${who}: ${kind} — ${item.fact.value}`
  }
  return item.label
}

/** A finished Update state result: tick the changes to apply. */
function StateProposal({ job }: { job: WritingJob }) {
  const result = job.result?.kind === 'state' ? job.result : null
  const bible = useProject((s) => s.state?.bible ?? [])
  const dismiss = useJobs((s) => s.dismiss)
  const [picked, setPicked] = useState<Set<string>>(() => new Set(result?.items.map((i) => i.key) ?? []))
  const [busy, setBusy] = useState(false)
  /** Reasons for the changes the computer did not apply; while shown, the card stays until it is closed. */
  const [rejected, setRejected] = useState<string[] | null>(null)
  if (!result) return null

  const apply = async (): Promise<void> => {
    setBusy(true)
    try {
      const res = await link.invoke('writing:applyState', job.id, [...picked])
      useProject.getState().setState(res.state)
      if (res.rejected.length > 0) {
        setRejected(res.rejected.map((r) => r.reason))
        toast.error(`${res.rejected.length} of the changes weren’t applied. The rest were.`)
        return
      }
      toast.success('The tracked state is updated.')
      await dismiss(job.id)
    } catch (e) {
      // For example the scene's text changed on the computer since the check was made.
      toast.error(errorMessage(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <JobShell job={job}>
      {result.items.length === 0 && <p className="text-[14px] text-muted">The AI found nothing new to record.</p>}
      <ul className="space-y-2">
        {result.items.map((item) => {
          const on = picked.has(item.key)
          return (
            <li key={item.key}>
              <button
                className="flex w-full items-start gap-2.5 rounded-lg border border-border bg-panel-2 p-2.5 text-left"
                onClick={() =>
                  setPicked((p) => {
                    const n = new Set(p)
                    if (n.has(item.key)) n.delete(item.key)
                    else n.add(item.key)
                    return n
                  })
                }
              >
                <span className={`mt-0.5 flex size-5 shrink-0 items-center justify-center rounded border ${on ? 'border-accent bg-accent text-accent-fg' : 'border-border-strong'}`}>{on && <Check size={13} strokeWidth={3} />}</span>
                <span className="min-w-0 text-[14px]">
                  <span className="block text-[11.5px] font-medium tracking-wide text-faint uppercase">{item.op === 'add' ? 'Add' : 'End'}</span>
                  {itemText(item, bible)}
                  {item.reason && <span className="mt-0.5 block text-[12.5px] text-muted">{item.reason}</span>}
                </span>
              </button>
            </li>
          )
        })}
      </ul>
      {result.left.length > 0 && (
        <div className="text-[12.5px] text-warn">
          <p className="font-medium">Left out:</p>
          <ul className="list-disc pl-5">
            {result.left.map((l, i) => (
              <li key={i}>{l}</li>
            ))}
          </ul>
        </div>
      )}
      {rejected && (
        <div className="text-[13px] text-danger">
          <p className="font-medium">Not applied:</p>
          <ul className="list-disc pl-5">
            {rejected.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}
      <div className="flex justify-end gap-2">
        <Button variant="ghost" disabled={busy} onClick={() => void dismiss(job.id)}>
          {rejected ? 'Close' : 'Reject'}
        </Button>
        <Button variant="primary" disabled={busy || picked.size === 0 || !!rejected} onClick={() => void apply()}>
          {busy ? <Spinner /> : <Check size={16} />} Apply {picked.size}
        </Button>
      </div>
    </JobShell>
  )
}

function AddFactSheet({ open, onOpenChange, sceneId }: { open: boolean; onOpenChange: (o: boolean) => void; sceneId: string }) {
  const bible = useProject((s) => s.state?.bible ?? [])
  const statePatch = useProject((s) => s.statePatch)
  const [entity, setEntity] = useState('')
  const [kind, setKind] = useState<StateKind>('location')
  const [value, setValue] = useState('')
  const [about, setAbout] = useState('')
  const fact = { entity: entity || bible[0]?.id || '', kind, value, about: about || null }
  // The other character is never the same entry as the one the fact is about.
  const withId = otherEntry(bible, fact.entity, about)
  const problem = value ? validateFact({ ...fact, about: kind === 'relationship' ? withId || null : null }) : null

  const save = async (): Promise<void> => {
    const f = { ...fact, about: kind === 'relationship' ? withId || null : null }
    const bad = validateFact(f)
    if (bad) return void toast.error(bad)
    const res = await statePatch(addFactOps(f, sceneId))
    if (res && res.rejected.length === 0) {
      setValue('')
      onOpenChange(false)
    }
  }

  return (
    <Sheet
      open={open}
      onOpenChange={onOpenChange}
      title="Add a fact"
      description="True from this scene on, until you end it."
      footer={
        <>
          <Button onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button variant="primary" disabled={!value.trim() || !!problem} onClick={() => void save()}>
            Add
          </Button>
        </>
      }
    >
      <div className="space-y-4 px-4 pb-4">
        <Field label="About">
          <Select value={fact.entity} onChange={(e) => setEntity(e.target.value)}>
            {bible.map((b) => (
              <option key={b.id} value={b.id}>
                {b.name}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Kind">
          <Select value={kind} onChange={(e) => setKind(e.target.value as StateKind)}>
            {STATE_KINDS.map((k) => (
              <option key={k.value} value={k.value}>
                {kindLabel(k.value)}
              </option>
            ))}
          </Select>
        </Field>
        {kind === 'relationship' && (
          <Field label="With">
            <Select value={withId} onChange={(e) => setAbout(e.target.value)}>
              {bible
                .filter((b) => b.id !== fact.entity)
                .map((b) => (
                  <option key={b.id} value={b.id}>
                    {b.name}
                  </option>
                ))}
            </Select>
          </Field>
        )}
        <Field label="Fact" hint="One line, for example “the lighthouse”.">
          <Input value={value} onChange={(e) => setValue(e.target.value)} maxLength={300} />
        </Field>
        {problem && <p className="text-[13.5px] text-warn">{problem}</p>}
      </div>
    </Sheet>
  )
}
