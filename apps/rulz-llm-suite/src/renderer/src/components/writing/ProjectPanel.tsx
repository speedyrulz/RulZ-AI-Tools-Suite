import * as Collapsible from '@radix-ui/react-collapsible'
import { ChevronRight, History as HistoryIcon, ListTree } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { NovelBudget } from '@shared/novel'
import { MAX_ADAPTER_SCALE, MIN_ADAPTER_SCALE } from '@shared/adapters'
import type { LoadAdapter, SamplingParams } from '@shared/types'
import { chatAdapters, formatScale, onScale, withChatScale } from '../../lib/adapters'
import { STYLE_PATH, useNovel } from '../../stores/novel'
import { useModels } from '../../stores/models'
import { VarietyFields } from '../chat/InferencePanel'
import { Button, Field, IconButton, NumberInput, Select, Slider, Switch } from '../ui'
import { BufferBanners } from './Banners'
import { CodeEditor } from './CodeEditor'
import { DraftInput, PanelSection } from './common'
import { HistoryDialog } from './HistoryDialog'

function SliderRow({ label, hint, value, min, max, step, onChange }: { label: string; hint?: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <Field label={label} hint={hint}>
      <div className="flex items-center gap-2.5">
        <Slider value={value} min={min} max={max} step={step} onChange={onChange} />
        <NumberInput className="w-18" value={value} min={min} max={max} step={step} onChange={(v) => onChange(Math.min(max, Math.max(min, v ?? min)))} />
      </div>
    </Field>
  )
}

export function ProjectPanel() {
  const project = useNovel((s) => s.state?.project)
  const blocked = useNovel((s) => !!s.state?.blocked)
  const busy = useNovel((s) => !!s.gen || !!s.pending)
  const models = useModels((s) => s.models)
  const instances = useModels((s) => s.instances)
  const activeModelId = useModels((s) => s.activeModelId)
  const [varietyOpen, setVarietyOpen] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [chapters, setChapters] = useState<number | null>(8)
  const [note, setNote] = useState('')
  // Sampling and budget are edited locally and saved after a short pause, so sliders stay smooth.
  const [sampling, setSampling] = useState<SamplingParams | null>(null)
  const [budget, setBudget] = useState<NovelBudget | null>(null)
  const timers = useRef<{ s?: ReturnType<typeof setTimeout>; b?: ReturnType<typeof setTimeout> }>({})

  useEffect(() => {
    setSampling(null)
    setBudget(null)
  }, [project?.id])
  useEffect(() => () => {
    const { s, b } = timers.current
    if (s) clearTimeout(s)
    if (b) clearTimeout(b)
  }, [])

  if (!project) return <div className="p-4 text-xs text-faint">Open a project to change its settings.</div>
  const store = useNovel.getState()
  const sp = sampling ?? project.sampling
  const bg = budget ?? project.budget
  const setS = <K extends keyof SamplingParams>(key: K, value: SamplingParams[K]): void => {
    const next = { ...sp, [key]: value }
    setSampling(next)
    if (timers.current.s) clearTimeout(timers.current.s)
    timers.current.s = setTimeout(() => void store.patchProject({ sampling: next }).then(() => setSampling(null)), 500)
  }
  const setB = (patch: Partial<NovelBudget>): void => {
    const next = { ...bg, ...patch }
    setBudget(next)
    if (timers.current.b) clearTimeout(timers.current.b)
    timers.current.b = setTimeout(() => void store.patchProject({ budget: next }).then(() => setBudget(null)), 500)
  }

  const modelId = project.modelId ?? activeModelId
  const inst = instances.find((i) => i.modelId === modelId && i.status === 'ready')
  const loaded = inst?.adapters ?? []
  const rows = chatAdapters(loaded, project.adapters)
  const setAdapter = (id: string, scale: number): void => void store.patchProject({ adapters: withChatScale(loaded, project.adapters, id, scale) as LoadAdapter[] })
  const varietyOn = (sp.dryMultiplier > 0 ? 1 : 0) + (sp.xtcProbability > 0 ? 1 : 0) + (sp.avoidWords.length > 0 ? 1 : 0)

  return (
    <div className={blocked ? 'pointer-events-none opacity-60' : undefined}>
      <PanelSection title="Project">
        <Field label="Title">
          <DraftInput value={project.title} onCommit={(v) => v.trim() && void store.patchProject({ title: v.trim() })} />
        </Field>
        <Field label="POV and tense" hint="Sent with the style">
          <DraftInput value={project.povTense} placeholder="Close third, past tense" onCommit={(v) => void store.patchProject({ povTense: v.trim() })} />
        </Field>
      </PanelSection>

      <PanelSection title="Model">
        <Select aria-label="Model" value={project.modelId ?? ''} onChange={(e) => void store.patchProject({ modelId: e.target.value || null })}>
          <option value="">Active chat model</option>
          {project.modelId && !models.some((m) => m.id === project.modelId) && <option value={project.modelId}>{project.modelId} (not found)</option>}
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name}
            </option>
          ))}
        </Select>
        <p className="text-[11px] text-faint">
          {project.modelId ? 'A model chosen here is loaded with its default settings when an action needs it.' : 'Actions use whichever model is active in Chat (Ctrl+L to load one).'}
        </p>
      </PanelSection>

      <PanelSection title="Sampling">
        <SliderRow label="Temperature" value={sp.temperature} min={0} max={2} step={0.01} onChange={(v) => setS('temperature', v)} />
        <SliderRow label="Min P" value={sp.minP} min={0} max={1} step={0.01} onChange={(v) => setS('minP', v)} />
        <SliderRow label="Top P" value={sp.topP} min={0} max={1} step={0.01} onChange={(v) => setS('topP', v)} />
        <SliderRow label="Top K" hint="0 = off" value={sp.topK} min={0} max={200} step={1} onChange={(v) => setS('topK', Math.round(v))} />
        <SliderRow label="Repeat penalty" value={sp.repeatPenalty} min={0.8} max={2} step={0.01} onChange={(v) => setS('repeatPenalty', v)} />
        <Field inline label="Seed" hint="Empty = random each time">
          <NumberInput value={sp.seed} placeholder="Random" onChange={(v) => setS('seed', v === null ? null : Math.round(v))} />
        </Field>
        <Collapsible.Root open={varietyOpen} onOpenChange={setVarietyOpen}>
          <Collapsible.Trigger className="flex w-full items-center gap-1 text-[11px] font-semibold tracking-wide text-faint uppercase hover:text-text">
            <ChevronRight size={14} className={varietyOpen ? 'rotate-90 transition-transform' : 'transition-transform'} />
            Repetition and variety
            {varietyOn > 0 && <span className="font-normal normal-case"> · {varietyOn} on</span>}
          </Collapsible.Trigger>
          <Collapsible.Content className="mt-3 space-y-3">
            <VarietyFields sampling={sp} onChange={setS} />
          </Collapsible.Content>
        </Collapsible.Root>
        <p className="text-[11px] text-faint">Thinking is turned off for prose and allowed for summaries and outlines.</p>
      </PanelSection>

      <PanelSection title="Adapters">
        {loaded.length === 0 ? (
          <p className="text-[11px] text-faint">{inst ? 'This model was loaded without adapters.' : 'The model is not loaded.'} Adapters are chosen when you load the model (Ctrl+L).</p>
        ) : (
          <>
            {rows.map((a) => (
              <Field key={a.id} label={<span className="break-all">{a.name}</span>} hint={<span className="tabular-nums">Default {formatScale(a.defaultScale)}</span>}>
                <div className="flex items-center gap-2.5">
                  <Switch checked={a.scale !== 0} onCheckedChange={(on) => setAdapter(a.id, on ? onScale(a) : 0)} />
                  <Slider value={Math.min(2, Math.max(0, a.scale))} min={0} max={2} step={0.05} onChange={(v) => setAdapter(a.id, v)} />
                  <NumberInput className="w-18" value={a.scale} min={MIN_ADAPTER_SCALE} max={MAX_ADAPTER_SCALE} step={0.05} onChange={(v) => setAdapter(a.id, v ?? a.defaultScale)} />
                </div>
              </Field>
            ))}
            <div className="flex items-start justify-between gap-2">
              <p className="text-[11px] text-faint">Strengths for this project; changing them makes the model read the prompt again.</p>
              <Button size="sm" variant="ghost" disabled={project.adapters === null} onClick={() => void store.patchProject({ adapters: null })}>
                Use model defaults
              </Button>
            </div>
          </>
        )}
      </PanelSection>

      <PanelSection title="Budget">
        <Field inline label="Reply size" hint="Tokens kept free for the answer">
          <NumberInput value={bg.replyTokens} min={64} step={128} onChange={(v) => setB({ replyTokens: Math.max(64, Math.round(v ?? 1024)) })} />
        </Field>
        <SliderRow label="Share for the text so far" hint={`${Math.round(bg.proseShare * 100)}%`} value={Math.round(bg.proseShare * 100)} min={10} max={80} step={5} onChange={(v) => setB({ proseShare: v / 100 })} />
        <SliderRow label="Share for bible entries" hint={`${Math.round(bg.bibleShare * 100)}%`} value={Math.round(bg.bibleShare * 100)} min={5} max={60} step={5} onChange={(v) => setB({ bibleShare: v / 100 })} />
      </PanelSection>

      <PanelSection
        title="Style (style.md)"
        action={
          <IconButton size="sm" label="History of style.md" onClick={() => setHistoryOpen(true)}>
            <HistoryIcon size={14} />
          </IconButton>
        }
      >
        <p className="text-[11px] text-faint">Voice, rules and anything else the model should follow. Sent with every request, so keep it short.</p>
        <StyleEditor readOnly={blocked} />
        <HistoryDialog path={STYLE_PATH} open={historyOpen} onOpenChange={setHistoryOpen} />
      </PanelSection>

      <PanelSection title="Outline">
        <p className="text-[11px] text-faint">Ask the model to propose chapters from the premise in your style and bible. Nothing is added until you accept.</p>
        <Field label="Note for the planner" hint="Premise, ending, length">
          <DraftInput value={note} onCommit={setNote} placeholder="A heist on a river ferry, five days, bitter ending" />
        </Field>
        <div className="flex items-center gap-2">
          <NumberInput className="w-14" value={chapters} min={1} max={60} step={1} onChange={(v) => setChapters(v === null ? null : Math.min(60, Math.max(1, Math.round(v))))} />
          <span className="text-[13px] text-muted">chapters</span>
          <div className="flex-1" />
          <Button size="sm" disabled={busy || !chapters} onClick={() => void store.planChapters(chapters ?? 8, note)}>
            <ListTree size={14} /> Plan chapters
          </Button>
        </div>
      </PanelSection>
    </div>
  )
}

function StyleEditor({ readOnly }: { readOnly: boolean }) {
  const buffer = useNovel((s) => s.buffers[STYLE_PATH])
  if (!buffer) return null
  return (
    <div className="overflow-hidden rounded-lg border border-border">
      <BufferBanners path={STYLE_PATH} />
      <div className="h-56 bg-bg">
        <CodeEditor path={STYLE_PATH} text={buffer.text} readOnly={readOnly || buffer.readOnly} prose={false} placeholder="Close third person, past tense. Plain, dry humour. No adverbs after dialogue." onChange={(t) => useNovel.getState().edit(STYLE_PATH, t)} />
      </div>
    </div>
  )
}
