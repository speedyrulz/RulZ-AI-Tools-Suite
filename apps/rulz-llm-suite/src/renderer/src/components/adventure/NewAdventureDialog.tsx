import { useEffect, useState } from 'react'
import type { AdventureSetup, AdventureStarter } from '@shared/adventure'
import type { NovelProjectSummary } from '@shared/novel'
import { cleanSetup, emptySetup, LENGTH_LABELS, setupFromStarter, setupProblem } from '../../lib/adventure'
import { loadDialogData, useAdventure } from '../../stores/adventure'
import { Button, Dialog, Field, Input, NumberInput, Select, Textarea } from '../ui'
import { Segmented } from './parts'

type Tab = 'idea' | 'project'
type ModelOption = { id: string; name: string; loaded: boolean }

export function NewAdventureDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const [tab, setTab] = useState<Tab>('idea')
  const [form, setForm] = useState<AdventureSetup>(emptySetup)
  const [folder, setFolder] = useState<string | null>(null)
  const [starters, setStarters] = useState<AdventureStarter[]>([])
  const [models, setModels] = useState<ModelOption[]>([])
  const [projects, setProjects] = useState<NovelProjectSummary[]>([])
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!open) return
    setTab('idea')
    setForm(emptySetup())
    setFolder(null)
    setBusy(false)
    let alive = true
    void loadDialogData().then((d) => {
      if (!alive) return
      setStarters(d.starters)
      setModels(d.models)
      setProjects(d.projects)
    })
    return () => {
      alive = false
    }
  }, [open])

  const patch = (p: Partial<AdventureSetup>): void => setForm((f) => ({ ...f, ...p }))
  const fromProject = tab === 'project'
  const problem = setupProblem(form, fromProject, folder)

  const pickProject = (f: string): void => {
    setFolder(f || null)
    const p = projects.find((x) => x.folder === f)
    if (p && !form.title.trim()) patch({ title: p.title })
  }

  const start = async (): Promise<void> => {
    if (problem || busy) return
    setBusy(true)
    const ok = await useAdventure.getState().startNew(cleanSetup(form), fromProject ? folder : null)
    setBusy(false)
    if (ok) onOpenChange(false)
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => !busy && onOpenChange(v)}
      title="New adventure"
      className="w-[min(640px,calc(100vw-32px))]"
      footer={
        <>
          {problem && <span className="mr-auto self-center text-xs text-faint">{problem}</span>}
          <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={busy}>
            Cancel
          </Button>
          <Button variant="primary" disabled={!!problem || busy} onClick={() => void start()}>
            {busy ? 'Starting…' : 'Start'}
          </Button>
        </>
      }
    >
      <div className="space-y-4 p-5">
        <Segmented
          label="Where the story comes from"
          value={tab}
          onChange={setTab}
          options={[
            { value: 'idea', label: 'Your own idea' },
            { value: 'project', label: 'From a Writing project' }
          ]}
        />

        {fromProject ? (
          <Field label="Writing project" hint="Its characters and places are used">
            <Select value={folder ?? ''} onChange={(e) => pickProject(e.target.value)}>
              <option value="">{projects.length ? 'Pick a project…' : 'No Writing projects yet'}</option>
              {projects.map((p) => (
                <option key={p.folder} value={p.folder}>
                  {p.title}
                </option>
              ))}
            </Select>
          </Field>
        ) : (
          starters.length > 0 && (
            <div className="space-y-1.5">
              <div className="text-[11px] font-semibold tracking-wide text-faint uppercase">Starter ideas</div>
              <div className="flex flex-wrap gap-1.5">
                {starters.map((s) => (
                  <Button key={s.id} size="sm" variant="secondary" onClick={() => setForm((f) => setupFromStarter(f, s))}>
                    {s.title}
                  </Button>
                ))}
              </div>
            </div>
          )
        )}

        <div className="grid grid-cols-2 gap-3">
          <Field label="Title">
            <Input value={form.title} onChange={(e) => patch({ title: e.target.value })} />
          </Field>
          <Field label="Genre">
            <Input value={form.genre} placeholder="Fantasy, noir mystery…" onChange={(e) => patch({ genre: e.target.value })} />
          </Field>
        </div>
        <Field label="Premise" hint={fromProject ? 'Optional: the project’s premise is used' : undefined}>
          <Textarea rows={3} value={form.premise} placeholder="What the story is about and where it starts" onChange={(e) => patch({ premise: e.target.value })} />
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Your character" hint="Optional">
            <Input value={form.character} placeholder="A name and a line about them" onChange={(e) => patch({ character: e.target.value })} />
          </Field>
          <Field label="Tone" hint="Optional">
            <Input value={form.tone} placeholder="Light and funny, grim, cosy…" onChange={(e) => patch({ tone: e.target.value })} />
          </Field>
        </div>
        <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
          <Field label="Passage length">
            <Segmented
              label="Passage length"
              value={form.length}
              onChange={(length) => patch({ length })}
              options={(['short', 'medium', 'long'] as const).map((v) => ({ value: v, label: LENGTH_LABELS[v] }))}
            />
          </Field>
          <Field label="Choices">
            <Segmented
              label="Choices after each passage"
              value={form.choiceCount}
              onChange={(choiceCount) => patch({ choiceCount })}
              options={[2, 3, 4].map((n) => ({ value: n, label: String(n) }))}
            />
          </Field>
        </div>
        <div className="grid grid-cols-[1fr_auto] items-end gap-3">
          <Field label="Model">
            <Select value={form.modelId ?? ''} onChange={(e) => patch({ modelId: e.target.value || null })}>
              <option value="">Loaded model</option>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                  {m.loaded ? ' (loaded)' : ''}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Temperature">
            <NumberInput value={form.temperature} onChange={(temperature) => patch({ temperature })} placeholder="Default" min={0} max={2} step={0.1} className="h-8.5 w-28" />
          </Field>
        </div>
      </div>
    </Dialog>
  )
}
