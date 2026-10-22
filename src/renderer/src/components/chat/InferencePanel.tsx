import { RotateCcw, Save, Trash2 } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import type { SamplingParams } from '@shared/types'
import { useApp } from '../../stores/app'
import { useChat } from '../../stores/chat'
import { useMcp } from '../../stores/mcp'
import { Button, Dialog, Field, IconButton, Input, NumberInput, Select, Slider, Switch, Textarea } from '../ui'

function SliderRow({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange
}: {
  label: string
  hint?: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
}) {
  return (
    <Field label={label} hint={hint}>
      <div className="flex items-center gap-2.5">
        <Slider value={value} min={min} max={max} step={step} onChange={onChange} />
        <NumberInput className="w-18" value={value} min={min} max={max} step={step} onChange={(v) => onChange(v ?? min)} />
      </div>
    </Field>
  )
}

function Section({ title, children, action }: { title: string; children: ReactNode; action?: ReactNode }) {
  return (
    <section className="space-y-3 border-b border-border px-4 py-4 last:border-b-0">
      <div className="flex items-center justify-between">
        <h3 className="text-[11px] font-semibold tracking-wide text-faint uppercase">{title}</h3>
        {action}
      </div>
      {children}
    </section>
  )
}

function ToolsSection() {
  const settings = useApp((s) => s.settings)
  const updateSettings = useApp((s) => s.updateSettings)
  const setPage = useApp((s) => s.setPage)
  const servers = useMcp((s) => s.servers)
  if (!settings) return null
  const disabled = new Set(settings.disabledTools)
  const connected = servers.filter((s) => s.state === 'connected')
  const tools = connected.flatMap((s) => s.tools).filter((t) => !disabled.has(t.qualifiedName)).length
  return (
    <Section
      title="Tools (MCP)"
      action={
        <button className="text-[11px] text-muted hover:text-accent" onClick={() => setPage('mcp')}>
          Manage
        </button>
      }
    >
      <Field inline label="Let the model use tools" hint={servers.length ? `${tools} tools from ${connected.length} of ${servers.length} servers` : 'No MCP servers configured'}>
        <Switch checked={settings.toolsEnabled} onCheckedChange={(v) => void updateSettings({ toolsEnabled: v })} />
      </Field>
    </Section>
  )
}

export function InferencePanel() {
  const current = useChat((s) => s.current)
  const presets = useChat((s) => s.presets)
  const mutate = useChat((s) => s.mutateCurrent)
  const { applyPreset, savePreset, deletePreset } = useChat.getState()
  const settings = useApp((s) => s.settings)
  const updateSettings = useApp((s) => s.updateSettings)
  const [saveOpen, setSaveOpen] = useState(false)
  const [presetName, setPresetName] = useState('')

  if (!current || !settings) {
    return <aside className="w-80 shrink-0 border-l border-border bg-panel p-4 text-xs text-faint">Open a chat to edit its settings.</aside>
  }

  const s = current.sampling
  const setS = <K extends keyof SamplingParams>(key: K, value: SamplingParams[K]): void =>
    mutate((c) => ({ ...c, sampling: { ...c.sampling, [key]: value } }))
  const activePreset = presets.find((p) => p.id === current.presetId)

  return (
    <aside className="flex w-80 shrink-0 flex-col overflow-y-auto border-l border-border bg-panel">
      <Section
        title="Preset"
        action={
          <div className="flex gap-0.5">
            <IconButton
              size="sm"
              label="Save as preset"
              onClick={() => {
                setPresetName(activePreset?.name ?? '')
                setSaveOpen(true)
              }}
            >
              <Save size={13} />
            </IconButton>
            {activePreset && (
              <IconButton size="sm" label="Delete preset" onClick={() => void deletePreset(activePreset.id)}>
                <Trash2 size={13} />
              </IconButton>
            )}
          </div>
        }
      >
        <Select value={current.presetId ?? ''} onChange={(e) => applyPreset(e.target.value || null)}>
          <option value="">No preset</option>
          {presets.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </Select>
      </Section>

      <Section title="System prompt">
        <Textarea
          rows={7}
          className="max-h-[40vh] min-h-24"
          placeholder="You are a helpful assistant…"
          value={current.systemPrompt}
          onChange={(e) => mutate((c) => ({ ...c, systemPrompt: e.target.value }))}
        />
      </Section>

      <Section
        title="Sampling"
        action={
          <IconButton size="sm" label="Reset to defaults" onClick={() => mutate((c) => ({ ...c, sampling: settings.defaultSampling }))}>
            <RotateCcw size={13} />
          </IconButton>
        }
      >
        <SliderRow label="Temperature" value={s.temperature} min={0} max={2} step={0.01} onChange={(v) => setS('temperature', v)} />
        <SliderRow label="Top K" hint="0 = off" value={s.topK} min={0} max={200} step={1} onChange={(v) => setS('topK', Math.round(v))} />
        <SliderRow label="Top P" value={s.topP} min={0} max={1} step={0.01} onChange={(v) => setS('topP', v)} />
        <SliderRow label="Min P" value={s.minP} min={0} max={1} step={0.01} onChange={(v) => setS('minP', v)} />
        <SliderRow label="Repeat penalty" value={s.repeatPenalty} min={0.8} max={2} step={0.01} onChange={(v) => setS('repeatPenalty', v)} />
        <SliderRow label="Presence penalty" value={s.presencePenalty} min={-2} max={2} step={0.05} onChange={(v) => setS('presencePenalty', v)} />
        <SliderRow label="Frequency penalty" value={s.frequencyPenalty} min={-2} max={2} step={0.05} onChange={(v) => setS('frequencyPenalty', v)} />
        <Field inline label="Max response tokens" hint="Empty = no limit">
          <NumberInput value={s.maxTokens} placeholder="∞" min={1} onChange={(v) => setS('maxTokens', v === null ? null : Math.max(1, Math.round(v)))} />
        </Field>
        <Field inline label="Seed" hint="Empty = random">
          <NumberInput value={s.seed} placeholder="Random" onChange={(v) => setS('seed', v === null ? null : Math.round(v))} />
        </Field>
        <Field label="Stop strings" hint="One per line">
          <Textarea
            rows={2}
            className="font-mono text-xs"
            value={s.stop.join('\n')}
            onChange={(e) => setS('stop', e.target.value.split('\n').filter((x) => x.length > 0))}
          />
        </Field>
      </Section>

      <ToolsSection />

      <Section title="Structured output">
        <Field inline label="Constrain to a JSON schema">
          <Switch
            checked={!!s.jsonSchema}
            onCheckedChange={(on) => setS('jsonSchema', on ? '{\n  "type": "object",\n  "properties": {}\n}' : '')}
          />
        </Field>
        {!!s.jsonSchema && (
          <Textarea rows={8} className="font-mono text-xs" value={s.jsonSchema} onChange={(e) => setS('jsonSchema', e.target.value)} />
        )}
      </Section>

      <Section title="Context">
        <Field label="When the chat outgrows the context">
          <Select
            value={settings.contextOverflow}
            onChange={(e) => void updateSettings({ contextOverflow: e.target.value as 'stop' | 'truncateMiddle' })}
          >
            <option value="truncateMiddle">Truncate middle (keep system prompt and first message)</option>
            <option value="stop">Stop and show an error</option>
          </Select>
        </Field>
      </Section>

      <Dialog
        open={saveOpen}
        onOpenChange={setSaveOpen}
        title="Save preset"
        description="Saves this chat's system prompt and sampling settings. Saving with an existing name replaces it."
        footer={
          <>
            <Button variant="ghost" onClick={() => setSaveOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="primary"
              disabled={!presetName.trim()}
              onClick={() => {
                void savePreset(presetName)
                setSaveOpen(false)
              }}
            >
              Save
            </Button>
          </>
        }
      >
        <div className="p-5">
          <Input autoFocus placeholder="Preset name" value={presetName} onChange={(e) => setPresetName(e.target.value)} />
        </div>
      </Dialog>
    </aside>
  )
}
