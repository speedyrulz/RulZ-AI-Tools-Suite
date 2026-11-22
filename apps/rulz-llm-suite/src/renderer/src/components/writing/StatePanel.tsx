import { Pencil, Plus, Trash2, TriangleAlert, X } from 'lucide-react'
import { useMemo, useState, type ReactNode } from 'react'
import type { BibleEntry, NovelOutline, StateFact, StateFactStatus, StateKind, StateOp } from '@shared/novel'
import { factStatus, scenePositions, stateAfter, stateBefore } from '@shared/novelState'
import { invoke } from '../../lib/api'
import { cn } from '../../lib/format'
import { factAddOp, factEditOps, factValues, type FactValues } from '../../lib/novelRun'
import { sceneOf, useNovel } from '../../stores/novel'
import { Badge, Button, Field, Input, Select } from '../ui'
import { ConfirmDialog, PanelSection } from './common'

export const KIND_NAMES: Record<StateKind, string> = {
  location: 'Location',
  knows: 'Knows',
  carries: 'Carries',
  condition: 'Condition',
  relationship: 'Relationship',
  other: 'Other'
}
const KINDS = Object.keys(KIND_NAMES) as StateKind[]

/** The facts that are not usable as they stand (their entry or scene is gone, or until comes before from). */
export function problemFacts(facts: StateFact[], outline: NovelOutline, bible: BibleEntry[]): { fact: StateFact; status: StateFactStatus }[] {
  const pos = scenePositions(outline)
  const ids = new Set(bible.map((e) => e.id))
  return facts.map((fact) => ({ fact, status: factStatus(fact, pos, ids) })).filter((x) => x.status !== 'ok')
}

/** How many facts need a look: shown in the tab label. */
export function useStateProblemCount(): number {
  const facts = useNovel((s) => s.state?.state?.facts)
  const outline = useNovel((s) => s.state?.outline)
  const bible = useNovel((s) => s.state?.bible)
  return useMemo(() => (facts && outline && bible ? problemFacts(facts, outline, bible).length : 0), [facts, outline, bible])
}

type View = 'before' | 'after' | 'all'

export function StatePanel() {
  const state = useNovel((s) => s.state)
  const target = useNovel((s) => s.target)
  const lastSceneId = useNovel((s) => s.lastSceneId)
  const gen = useNovel((s) => s.gen)
  const [view, setView] = useState<View>('before')
  const [notes, setNotes] = useState<string[]>([])
  const [editing, setEditing] = useState<string | null>(null)
  const [adding, setAdding] = useState(false)
  const [confirmReset, setConfirmReset] = useState(false)
  const file = state?.state
  const facts = file?.facts
  const outline = state?.outline
  const bible = state?.bible

  const sceneId = target?.kind === 'scene' ? target.id : lastSceneId
  const scene = sceneId ? sceneOf(state, sceneId)?.scene : null
  const shown = useMemo(() => {
    if (!facts || !outline || !bible) return []
    const pos = scenePositions(outline)
    const ids = new Set(bible.map((e) => e.id))
    if (view === 'all' || !scene) return facts.filter((f) => factStatus(f, pos, ids) === 'ok')
    return view === 'before' ? stateBefore(facts, scene.id, outline, bible) : stateAfter(facts, scene.id, outline, bible)
  }, [facts, outline, bible, view, scene])
  const problems = useMemo(() => (facts && outline && bible ? problemFacts(facts, outline, bible) : []), [facts, outline, bible])
  const groups = useMemo(() => {
    const by = new Map<string, StateFact[]>()
    for (const f of shown) by.set(f.entity, [...(by.get(f.entity) ?? []), f])
    return [...by.entries()]
  }, [shown])

  if (!state || !outline || !bible) return null
  const readOnly = !!state.blocked || !!file?.problem
  const why = state.blocked ? 'The project is read-only.' : file?.problem ? 'state.json cannot be used.' : undefined
  const scenes = outline.chapters.flatMap((c) => c.scenes.map((s) => ({ id: s.id, title: `${c.title}: ${s.title}` })))
  const nameOf = (id: string): string => bible.find((e) => e.id === id)?.name ?? `${id} (not in the bible)`
  const sceneTitle = (id: string): string => scenes.find((s) => s.id === id)?.title.split(': ').slice(1).join(': ') ?? `${id} (not in the outline)`

  const send = async (ops: StateOp[]): Promise<boolean> => {
    const r = await useNovel.getState().statePatch(ops)
    if (!r) return false
    setNotes(r.rejected.map((x) => x.reason))
    return r.rejected.length === 0
  }

  const row = (f: StateFact, status: StateFactStatus): ReactNode => {
    if (editing === f.id) {
      return (
        <li key={f.id} className="rounded-md border border-border-strong bg-panel p-2">
          <FactForm
            bible={bible}
            scenes={scenes}
            initial={factValues(f)}
            submit="Save"
            onCancel={() => setEditing(null)}
            onSubmit={async (v) => {
              const ops = factEditOps(f, v)
              if (!ops.length || (await send(ops))) setEditing(null)
            }}
          />
        </li>
      )
    }
    return (
      <li key={f.id} className="space-y-1 rounded-md border border-border bg-panel p-2 text-[13px]">
        <div className="flex flex-wrap items-center gap-1.5">
          <Badge>{KIND_NAMES[f.kind]}</Badge>
          {f.kind === 'relationship' && f.about && <span className="text-xs text-muted">with {nameOf(f.about)}</span>}
          {status === 'detached' && <Badge tone="warn">Detached</Badge>}
          {status === 'inverted' && <Badge tone="warn">Ends before it starts</Badge>}
        </div>
        <div className="text-body">{f.value}</div>
        <div className="text-[11px] text-faint">
          {status !== 'ok' && <>{nameOf(f.entity)}. </>}from {sceneTitle(f.from)}
          {f.until ? `, until ${sceneTitle(f.until)}` : ''}
        </div>
        <div className="flex flex-wrap justify-end gap-1">
          {status === 'ok' && !f.until && scene && (
            <Button size="sm" variant="ghost" disabled={readOnly} title={why ?? `Stop this being true from "${scene.title}" on`} onClick={() => void send([{ op: 'end', factId: f.id, until: scene.id }])}>
              End here
            </Button>
          )}
          <Button size="sm" variant="ghost" disabled={readOnly} title={why} aria-label={`Edit: ${f.value}`} onClick={() => setEditing(f.id)}>
            <Pencil size={13} /> Edit
          </Button>
          <Button size="sm" variant="ghost" disabled={readOnly} title={why} aria-label={`Remove: ${f.value}`} onClick={() => void send([{ op: 'remove', factId: f.id }])}>
            <Trash2 size={13} /> Remove
          </Button>
        </div>
      </li>
    )
  }

  return (
    <div>
      {file?.problem && (
        <PanelSection title="State file">
          <p className="flex items-start gap-1.5 text-[13px] text-danger-text" role="alert">
            <TriangleAlert size={15} className="mt-0.5 shrink-0" /> {file.problem} It is shown read-only and nothing is written to it.
          </p>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" onClick={() => void invoke('shell:showItem', state.folder)}>
              Show in folder
            </Button>
            {file.canReset && (
              <Button size="sm" disabled={!!state.blocked} onClick={() => setConfirmReset(true)}>
                Start empty state
              </Button>
            )}
          </div>
        </PanelSection>
      )}

      <PanelSection
        title="Tracked state"
        action={
          <Button size="sm" disabled={readOnly || !!gen} title={why ?? (scene ? `Record changes in "${scene.title}"` : 'Open a scene first')} onClick={() => scene && void useNovel.getState().updateState(scene.id)}>
            Update state
          </Button>
        }
      >
        <div className="flex gap-1" role="group" aria-label="Which facts">
          {(['before', 'after', 'all'] as View[]).map((v) => (
            <button
              key={v}
              aria-pressed={view === v}
              onClick={() => setView(v)}
              className={cn('h-7 flex-1 rounded-md border px-2 text-xs', view === v ? 'border-border-strong bg-hover text-text' : 'border-border text-muted hover:bg-hover')}
            >
              {v === 'before' ? 'Before scene' : v === 'after' ? 'After scene' : 'All'}
            </button>
          ))}
        </div>
        <p className="text-[11px] text-faint">{view === 'all' || !scene ? 'Every usable fact in the book.' : `What holds ${view === 'before' ? 'before' : 'after'} "${scene.title}".`}</p>
        {notes.length > 0 && (
          <div className="text-xs text-danger-text" role="alert">
            Not applied:
            <ul className="mt-0.5 list-disc pl-4">
              {notes.map((n, i) => (
                <li key={i}>{n}</li>
              ))}
            </ul>
          </div>
        )}
        {groups.length === 0 ? (
          <p className="text-xs text-faint">No facts here yet. Use Update state after writing a scene, or add one below.</p>
        ) : (
          groups.map(([entity, list]) => (
            <div key={entity} className="space-y-1.5">
              <h4 className="text-[13px] font-medium">{nameOf(entity)}</h4>
              <ul className="space-y-1.5">{list.map((f) => row(f, 'ok'))}</ul>
            </div>
          ))
        )}
      </PanelSection>

      {problems.length > 0 && (
        <PanelSection title={`Needs a look (${problems.length})`}>
          <p className="text-[11px] text-faint">These point at an entry or scene that is gone, or end before they start. They are not used in requests until you fix or remove them.</p>
          <ul className="space-y-1.5">{problems.map((p) => row(p.fact, p.status))}</ul>
        </PanelSection>
      )}

      {file && file.invalid.length > 0 && (
        <PanelSection title={`Could not be read (${file.invalid.length})`}>
          <p className="text-[11px] text-faint">These entries in state.json are kept as they are. Remove one only if you do not need it.</p>
          <ul className="space-y-1.5">
            {file.invalid.map((x) => (
              <li key={x.hash} className="space-y-1 rounded-md border border-border bg-panel p-2 text-xs">
                <div className="text-warn-text">{x.reason}</div>
                <pre className="max-h-24 overflow-auto font-mono text-[11px] whitespace-pre-wrap text-muted">{x.raw}</pre>
                <div className="flex justify-end">
                  <Button size="sm" variant="ghost" disabled={readOnly} title={why} onClick={() => void send([{ op: 'removeInvalid', index: x.index, hash: x.hash }])}>
                    <X size={13} /> Remove
                  </Button>
                </div>
              </li>
            ))}
          </ul>
        </PanelSection>
      )}

      <PanelSection title="Add a fact">
        {adding ? (
          <FactForm
            bible={bible}
            scenes={scenes}
            initial={{ entity: scene?.pov ?? bible.find((e) => e.kind === 'character')?.id ?? bible[0]?.id ?? '', kind: 'location', value: '', about: '', from: scene?.id ?? scenes[0]?.id ?? '', until: '' }}
            submit="Add"
            onCancel={() => setAdding(false)}
            onSubmit={async (v) => {
              if (await send([factAddOp(v)])) setAdding(false)
            }}
          />
        ) : (
          <Button size="sm" disabled={readOnly || bible.length === 0 || scenes.length === 0} title={why ?? (bible.length === 0 ? 'Add a bible entry first' : undefined)} onClick={() => setAdding(true)}>
            <Plus size={14} /> Add fact
          </Button>
        )}
      </PanelSection>

      <ConfirmDialog
        open={confirmReset}
        onOpenChange={setConfirmReset}
        title="Start an empty state?"
        danger={false}
        body={<p>state.json is missing or empty. This writes a new empty one, so facts can be recorded again.</p>}
        confirm="Start empty state"
        onConfirm={() => void send([{ op: 'reset' }])}
      />
    </div>
  )
}

function FactForm({
  bible,
  scenes,
  initial,
  submit,
  onSubmit,
  onCancel
}: {
  bible: BibleEntry[]
  scenes: { id: string; title: string }[]
  initial: FactValues
  submit: string
  onSubmit: (v: FactValues) => void | Promise<void>
  onCancel: () => void
}) {
  const [v, setV] = useState(initial)
  const set = (patch: Partial<FactValues>): void => setV((x) => ({ ...x, ...patch }))
  const valid = !!v.entity && !!v.from && !!v.value.trim() && v.value.trim().length <= 300 && (v.kind !== 'relationship' || !!v.about)
  const entityOptions = bible.some((e) => e.id === v.entity) ? bible : [...bible, { id: v.entity, name: `${v.entity} (not in the bible)` } as BibleEntry]
  const sceneOptions = (id: string): { id: string; title: string }[] => (scenes.some((s) => s.id === id) || !id ? scenes : [...scenes, { id, title: `${id} (not in the outline)` }])
  return (
    <form
      className="space-y-2"
      onSubmit={(e) => {
        e.preventDefault()
        if (valid) void onSubmit(v)
      }}
    >
      <Field label="About">
        <Select aria-label="Entity" value={v.entity} onChange={(e) => set({ entity: e.target.value })}>
          {entityOptions.map((e) => (
            <option key={e.id} value={e.id}>
              {e.name}
            </option>
          ))}
        </Select>
      </Field>
      <Field label="Kind">
        <Select aria-label="Kind" value={v.kind} onChange={(e) => set({ kind: e.target.value as StateKind })}>
          {KINDS.map((k) => (
            <option key={k} value={k}>
              {KIND_NAMES[k]}
            </option>
          ))}
        </Select>
      </Field>
      {v.kind === 'relationship' && (
        <Field label="With">
          <Select aria-label="Other entry" value={v.about} onChange={(e) => set({ about: e.target.value })}>
            <option value="">Choose…</option>
            {(v.about && !bible.some((e) => e.id === v.about) ? [...bible, { id: v.about, name: `${v.about} (not in the bible)` } as BibleEntry] : bible).map((e) => (
              <option key={e.id} value={e.id}>
                {e.name}
              </option>
            ))}
          </Select>
        </Field>
      )}
      <Field label="Fact" hint="One line, up to 300 characters.">
        <Input aria-label="Value" value={v.value} maxLength={300} onChange={(e) => set({ value: e.target.value })} />
      </Field>
      <Field label="True from">
        <Select aria-label="From scene" value={v.from} onChange={(e) => set({ from: e.target.value })}>
          {sceneOptions(v.from).map((s) => (
            <option key={s.id} value={s.id}>
              {s.title}
            </option>
          ))}
        </Select>
      </Field>
      <Field label="Until">
        <Select aria-label="Until scene" value={v.until} onChange={(e) => set({ until: e.target.value })}>
          <option value="">Still true</option>
          {sceneOptions(v.until).map((s) => (
            <option key={s.id} value={s.id}>
              {s.title}
            </option>
          ))}
        </Select>
      </Field>
      <div className="flex justify-end gap-2">
        <Button size="sm" type="button" variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
        <Button size="sm" type="submit" variant="primary" disabled={!valid}>
          {submit}
        </Button>
      </div>
    </form>
  )
}
