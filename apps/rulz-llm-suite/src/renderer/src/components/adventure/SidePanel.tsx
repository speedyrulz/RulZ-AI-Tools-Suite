import { Backpack, Map as MapIcon, Pencil, Sliders } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import type { AdventureSetup } from '@shared/adventure'
import { LENGTH_LABELS, mapLabel, mapRows } from '../../lib/adventure'
import { invoke } from '../../lib/api'
import { cn } from '../../lib/format'
import { useAdventure, type PanelTab } from '../../stores/adventure'
import { Button, Field, Input, NumberInput, Select } from '../ui'
import { FOCUS, Segmented, StateView } from './parts'

const TABS: { id: PanelTab; label: string; icon: typeof Backpack }[] = [
  { id: 'state', label: 'State', icon: Backpack },
  { id: 'map', label: 'Map', icon: MapIcon },
  { id: 'settings', label: 'Settings', icon: Sliders }
]

export function AdventureSidePanel({ onEdit }: { onEdit: (nodeId: string) => void }) {
  const tab = useAdventure((s) => s.panelTab)
  const setTab = useAdventure((s) => s.setPanelTab)
  return (
    <aside aria-label="Adventure panel" className="flex w-80 shrink-0 flex-col border-l border-border bg-panel">
      <div role="tablist" className="flex gap-1 border-b border-border p-2">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={cn(
              'flex h-8 flex-1 items-center justify-center gap-1.5 rounded-md text-[13px] transition-colors',
              FOCUS,
              tab === t.id ? 'bg-hover font-medium text-text' : 'text-muted hover:bg-hover hover:text-text'
            )}
          >
            <t.icon size={14} aria-hidden /> {t.label}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {tab === 'state' && <StateTab onEdit={onEdit} />}
        {tab === 'map' && <MapTab />}
        {tab === 'settings' && <SettingsTab />}
      </div>
    </aside>
  )
}

function StateTab({ onEdit }: { onEdit: (nodeId: string) => void }) {
  const node = useAdventure((s) => (s.position ? s.adv?.nodes[s.position] : undefined))
  if (!node) return <p className="text-[13px] text-faint">The state shows once the opening is written.</p>
  return (
    <div className="space-y-4">
      <StateView state={node.state} />
      <Button size="sm" onClick={() => onEdit(node.id)}>
        <Pencil size={14} aria-hidden /> Edit
      </Button>
    </div>
  )
}

function MapTab() {
  const adv = useAdventure((s) => s.adv)
  const position = useAdventure((s) => s.position)
  const go = useAdventure((s) => s.go)
  const rows = useMemo(() => (adv ? mapRows(adv, position) : []), [adv, position])
  if (!rows.length) return <p className="text-[13px] text-faint">The map shows once the opening is written.</p>
  return (
    <ul className="space-y-0.5">
      {rows.map((r) => (
        <li key={r.node.id} style={{ paddingLeft: Math.min(r.depth, 12) * 12 }}>
          <button
            onClick={() => go(r.node.id)}
            aria-current={r.isPosition ? 'true' : undefined}
            title={r.node.input.kind === 'start' ? 'The opening' : r.node.input.text}
            className={cn(
              'block w-full truncate rounded-md px-2 py-1 text-left text-[13px]',
              FOCUS,
              r.isPosition ? 'bg-accent-soft text-accent-text' : r.onPath ? 'text-text hover:bg-hover' : 'text-muted hover:bg-hover hover:text-text'
            )}
          >
            {mapLabel(r.node)}
            {r.node.ended && <span className="ml-1.5 text-[11px] text-faint">end</span>}
          </button>
        </li>
      ))}
    </ul>
  )
}

/** A text field that keeps what is typed and saves on blur or Enter. */
function DraftInput({ value, onCommit, label, required }: { value: string; onCommit: (v: string) => void; label: string; required?: boolean }) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])
  const commit = (): void => {
    const v = draft.trim()
    if (v === value || (required && !v)) return setDraft(value)
    onCommit(v)
  }
  return <Input aria-label={label} value={draft} onChange={(e) => setDraft(e.target.value)} onBlur={commit} onKeyDown={(e) => e.key === 'Enter' && commit()} />
}

function SettingsTab() {
  const adv = useAdventure((s) => s.adv)
  const update = useAdventure((s) => s.updateSetup)
  const [models, setModels] = useState<{ id: string; name: string; loaded: boolean }[]>([])
  useEffect(() => {
    let alive = true
    void invoke('adventure:models').then((m) => alive && setModels(m), () => undefined)
    return () => {
      alive = false
    }
  }, [])
  if (!adv) return null
  const s = adv.setup
  const set = (p: Partial<Omit<AdventureSetup, 'world' | 'premise' | 'character'>>): void => void update(p)
  const modelKnown = s.modelId === null || models.some((m) => m.id === s.modelId)
  return (
    <div className="space-y-4">
      <Field label="Title">
        <DraftInput label="Title" value={s.title} required onCommit={(title) => set({ title })} />
      </Field>
      <Field label="Genre">
        <DraftInput label="Genre" value={s.genre} onCommit={(genre) => set({ genre })} />
      </Field>
      <Field label="Tone">
        <DraftInput label="Tone" value={s.tone} onCommit={(tone) => set({ tone })} />
      </Field>
      <Field label="Passage length">
        <Segmented label="Passage length" value={s.length} onChange={(length) => set({ length })} options={(['short', 'medium', 'long'] as const).map((v) => ({ value: v, label: LENGTH_LABELS[v] }))} />
      </Field>
      <Field label="Choices">
        <Segmented label="Choices after each passage" value={s.choiceCount} onChange={(choiceCount) => set({ choiceCount })} options={[2, 3, 4].map((n) => ({ value: n, label: String(n) }))} />
      </Field>
      <Field label="Model">
        <Select value={s.modelId ?? ''} onChange={(e) => set({ modelId: e.target.value || null })}>
          <option value="">Loaded model (not OpenRouter)</option>
          {!modelKnown && <option value={s.modelId ?? ''}>{s.modelId}</option>}
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name}
              {m.loaded ? ' (loaded)' : ''}
            </option>
          ))}
        </Select>
      </Field>
      <Field label="Temperature" hint="Empty = default">
        <NumberInput value={s.temperature} onChange={(temperature) => set({ temperature })} placeholder="Default" min={0} max={2} step={0.1} className="h-8.5 w-28" />
      </Field>
      {s.world && (
        <Field label={`From the Writing project “${s.world.title}”`} hint="Read only">
          <pre className="max-h-56 overflow-y-auto rounded-lg border border-border bg-bg p-2.5 text-[12px] leading-relaxed whitespace-pre-wrap text-muted">{s.world.text}</pre>
        </Field>
      )}
    </div>
  )
}
