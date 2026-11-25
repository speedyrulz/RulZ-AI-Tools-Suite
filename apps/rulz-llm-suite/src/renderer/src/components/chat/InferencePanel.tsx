import * as Collapsible from '@radix-ui/react-collapsible'
import { ChevronRight, RotateCcw, Save, Trash2 } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { useShallow } from 'zustand/react/shallow'
import { breakersToText, textToBreakers, textToWords } from '@shared/variety'
import { modeSampling } from '@shared/modes'
import type { SamplerSupport } from '@shared/cloud'
import { REASONING_EFFORTS, type ReasoningEffort, type SamplingParams } from '@shared/types'
import { useApp } from '../../stores/app'
import { useChat } from '../../stores/chat'
import { modeDefaultsOf, useModels } from '../../stores/models'
import { effectiveToolMode, findTool, tokensOf, useMcp } from '../../stores/mcp'
import { NOT_SUPPORTED, supportsOf } from '../../lib/cloud'
import { formatCompact } from '../../lib/format'
import { Button, Dialog, Field, IconButton, Input, NumberInput, Select, Slider, Switch, Textarea } from '../ui'
import { AdapterPanel } from './AdapterPanel'

function SliderRow({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
  off
}: {
  label: string
  hint?: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
  /** The model does not honour this setting: shown but disabled. */
  off?: boolean
}) {
  return (
    <Field label={label} hint={off ? NOT_SUPPORTED : hint}>
      <div className="flex items-center gap-2.5">
        <Slider value={value} min={min} max={max} step={step} onChange={onChange} disabled={off} />
        {/* Typed values keep to the slider's range: a repeat penalty of 0, say, wrecks the output. */}
        <NumberInput className="w-18" value={value} min={min} max={max} step={step} disabled={off} onChange={(v) => onChange(Math.min(max, Math.max(min, v ?? min)))} />
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

/**
 * The core sampling controls of the chat panel; also used for a model's per-mode defaults. `limits` = max tokens, seed
 * and stop strings, which the Adventure and Writing modes set themselves. `supports` (an OpenRouter model) disables what the model
 * does not honour; undefined = every control is live.
 */
export function SamplingFields({
  sampling: s,
  onChange: setS,
  limits = true,
  supports
}: {
  sampling: SamplingParams
  onChange: <K extends keyof SamplingParams>(key: K, value: SamplingParams[K]) => void
  limits?: boolean
  supports?: SamplerSupport
}) {
  const off = (k: keyof SamplerSupport): boolean => !!supports && !supports[k]
  return (
    <>
      <SliderRow label="Temperature" value={s.temperature} min={0} max={2} step={0.01} off={off('temperature')} onChange={(v) => setS('temperature', v)} />
      <SliderRow label="Top K" hint="0 = off" value={s.topK} min={0} max={200} step={1} off={off('topK')} onChange={(v) => setS('topK', Math.round(v))} />
      <SliderRow label="Top P" value={s.topP} min={0} max={1} step={0.01} off={off('topP')} onChange={(v) => setS('topP', v)} />
      <SliderRow label="Min P" value={s.minP} min={0} max={1} step={0.01} off={off('minP')} onChange={(v) => setS('minP', v)} />
      <SliderRow label="Repeat penalty" value={s.repeatPenalty} min={0.8} max={2} step={0.01} off={off('repeatPenalty')} onChange={(v) => setS('repeatPenalty', v)} />
      <SliderRow label="Presence penalty" value={s.presencePenalty} min={-2} max={2} step={0.05} off={off('presencePenalty')} onChange={(v) => setS('presencePenalty', v)} />
      <SliderRow label="Frequency penalty" value={s.frequencyPenalty} min={-2} max={2} step={0.05} off={off('frequencyPenalty')} onChange={(v) => setS('frequencyPenalty', v)} />
      {limits && (
        <>
          <Field inline label="Max response tokens" hint="Empty = no limit">
            <NumberInput value={s.maxTokens} placeholder="∞" min={1} onChange={(v) => setS('maxTokens', v === null ? null : Math.max(1, Math.round(v)))} />
          </Field>
          <Field inline label="Seed" hint={off('seed') ? NOT_SUPPORTED : 'Empty = random'}>
            <NumberInput value={s.seed} placeholder="Random" disabled={off('seed')} onChange={(v) => setS('seed', v === null ? null : Math.round(v))} />
          </Field>
          <Field label="Stop strings" hint={off('stop') ? NOT_SUPPORTED : 'One per line'}>
            <Textarea
              rows={2}
              className="font-mono text-xs disabled:pointer-events-none disabled:opacity-45"
              disabled={off('stop')}
              value={s.stop.join('\n')}
              onChange={(e) => setS('stop', e.target.value.split('\n').filter((x) => x.length > 0))}
            />
          </Field>
        </>
      )}
    </>
  )
}

const EFFORT_LABELS: Record<ReasoningEffort, string> = {
  default: "Model's default",
  minimal: 'Minimal',
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  xhigh: 'Extra high',
  max: 'Max'
}

/** Thinking-mode controls; also used for the new-chat defaults on the Settings page. `supports` as for SamplingFields. */
export function ThinkingFields({
  sampling: s,
  onChange,
  supports
}: {
  sampling: SamplingParams
  onChange: <K extends keyof SamplingParams>(key: K, value: SamplingParams[K]) => void
  supports?: SamplerSupport
}) {
  const noThinking = !!supports && !supports.thinking
  const noEffort = !!supports && !supports.reasoningEffort
  const noBudget = !!supports && !supports.thinkingBudget
  return (
    <>
      <Field inline label="Thinking" hint={noThinking ? NOT_SUPPORTED : supports && !supports.thinkingOff ? "This model always thinks, so Off is not available" : "For reasoning models; Auto keeps the model's own behaviour"}>
        <Select className="w-28" value={s.thinking} disabled={noThinking} onChange={(e) => onChange('thinking', e.target.value as SamplingParams['thinking'])}>
          <option value="auto">Auto</option>
          <option value="on">On</option>
          <option value="off" disabled={!!supports && !supports.thinkingOff}>
            Off
          </option>
        </Select>
      </Field>
      {s.thinking !== 'off' && (
        <>
          <Field inline label="Reasoning effort" hint={noEffort ? NOT_SUPPORTED : 'Only models whose template takes an effort level (e.g. gpt-oss)'}>
            <Select className="w-32" value={s.reasoningEffort} disabled={noEffort} onChange={(e) => onChange('reasoningEffort', e.target.value as ReasoningEffort)}>
              {REASONING_EFFORTS.map((r) => (
                <option key={r} value={r}>
                  {EFFORT_LABELS[r]}
                </option>
              ))}
            </Select>
          </Field>
          <Field inline label="Thinking budget" hint={noBudget ? NOT_SUPPORTED : 'Tokens before the answer is forced; empty = unlimited'}>
            <NumberInput
              value={s.thinkingBudget}
              placeholder="∞"
              min={0}
              step={256}
              disabled={noBudget}
              onChange={(v) => onChange('thinkingBudget', v === null ? null : Math.max(0, Math.round(v)))}
            />
          </Field>
          {s.thinkingBudget !== null && (
            <Field label="Budget message" hint="Added when the budget runs out">
              <Input
                placeholder="Okay, time to give the answer."
                value={s.thinkingBudgetMessage}
                onChange={(e) => onChange('thinkingBudgetMessage', e.target.value)}
              />
            </Field>
          )}
        </>
      )}
    </>
  )
}

/** A one-per-line list. Keeps its own text while typing, so a new empty line stays until it is filled. */
function ListArea({
  value,
  parse,
  format,
  placeholder,
  onChange,
  disabled
}: {
  value: string[]
  parse: (text: string) => string[]
  format: (list: string[]) => string
  placeholder?: string
  onChange: (list: string[]) => void
  disabled?: boolean
}) {
  const [draft, setDraft] = useState(() => format(value))
  const [seen, setSeen] = useState(value)
  // Another source changed the list (a preset, a reset): show it.
  if (seen !== value) {
    setSeen(value)
    if (JSON.stringify(parse(draft)) !== JSON.stringify(value)) setDraft(format(value))
  }
  return (
    <Textarea
      rows={3}
      className="font-mono text-xs disabled:pointer-events-none disabled:opacity-45"
      placeholder={placeholder}
      disabled={disabled}
      value={draft}
      onChange={(e) => {
        setDraft(e.target.value)
        onChange(parse(e.target.value))
      }}
    />
  )
}

/** Repetition and variety controls (DRY, XTC, avoid words); also used for the new-chat defaults on the Settings page. */
export function VarietyFields({
  sampling: s,
  onChange,
  supports
}: {
  sampling: SamplingParams
  onChange: <K extends keyof SamplingParams>(key: K, value: SamplingParams[K]) => void
  supports?: SamplerSupport
}) {
  const noDry = !!supports && !supports.dry
  const noXtc = !!supports && !supports.xtc
  const noAvoid = !!supports && !supports.avoidWords
  return (
    <>
      <SliderRow
        off={noDry}
        label="DRY multiplier"
        hint="0 = off. Penalises text that would repeat an earlier passage; around 0.8 is typical"
        value={s.dryMultiplier}
        min={0}
        max={3}
        step={0.05}
        onChange={(v) => onChange('dryMultiplier', v)}
      />
      {s.dryMultiplier > 0 && !noDry && (
        <>
          <SliderRow label="DRY base" hint="How fast the penalty grows with the length of the repeat" value={s.dryBase} min={1} max={4} step={0.05} onChange={(v) => onChange('dryBase', v)} />
          <Field inline label="DRY allowed length" hint="Repeats up to this many tokens are not penalised">
            <NumberInput value={s.dryAllowedLength} min={1} max={100} step={1} onChange={(v) => onChange('dryAllowedLength', Math.max(1, Math.round(v ?? 2)))} />
          </Field>
          <Field inline label="DRY look-back" hint="Tokens to look back over; -1 = the whole context, 0 = off">
            <NumberInput value={s.dryPenaltyLastN} min={-1} step={64} onChange={(v) => onChange('dryPenaltyLastN', Math.max(-1, Math.round(v ?? -1)))} />
          </Field>
          <Field label="DRY sequence breakers" hint="One per line. A breaker ends a repeat, so DRY does not penalise what follows it. Add character names here to stop DRY from avoiding them. Write a newline as \n">
            <ListArea value={s.drySequenceBreakers} parse={textToBreakers} format={breakersToText} onChange={(l) => onChange('drySequenceBreakers', l)} />
          </Field>
        </>
      )}
      <SliderRow
        off={noXtc}
        label="XTC probability"
        hint="0 = off. How often the likeliest tokens are skipped for fresher wording; around 0.5 is typical"
        value={s.xtcProbability}
        min={0}
        max={1}
        step={0.05}
        onChange={(v) => onChange('xtcProbability', v)}
      />
      {s.xtcProbability > 0 && !noXtc && (
        <SliderRow
          label="XTC threshold"
          hint="Only tokens more likely than this can be skipped; above 0.5 turns XTC off"
          value={s.xtcThreshold}
          min={0}
          max={0.6}
          step={0.01}
          onChange={(v) => onChange('xtcThreshold', v)}
        />
      )}
      <Field
        label="Avoid words"
        hint={noAvoid ? NOT_SUPPORTED : 'One per line. Works on tokens, so use distinctive words such as tapestry or testament. A phrase made of common words would penalise every word in it'}
      >
        <ListArea value={s.avoidWords} parse={textToWords} format={(l) => l.join('\n')} disabled={noAvoid} onChange={(l) => onChange('avoidWords', l)} />
      </Field>
      {s.avoidWords.length > 0 && !noAvoid && (
        <SliderRow
          label="Avoid strength"
          hint="How hard to push these words down; 100 = effectively banned"
          value={s.avoidStrength}
          min={1}
          max={100}
          step={1}
          onChange={(v) => onChange('avoidStrength', Math.round(v))}
        />
      )}
    </>
  )
}

const NO_TOOLS: string[] = []

function ToolsSection() {
  const settings = useApp((s) => s.settings)
  const updateSettings = useApp((s) => s.updateSettings)
  const setPage = useApp((s) => s.setPage)
  const servers = useMcp((s) => s.servers)
  const activeTools = useChat((s) => s.current?.activeTools ?? NO_TOOLS)
  const mutate = useChat((s) => s.mutateCurrent)
  if (!settings) return null
  const disabled = new Set(settings.disabledTools)
  const connected = servers.filter((s) => s.state === 'connected')
  const tools = connected.flatMap((s) => s.tools).filter((t) => !disabled.has(t.qualifiedName)).length
  const mode = effectiveToolMode()
  const loaded = activeTools.filter((n) => findTool(n))
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
      {mode && (
        <div className="text-[11px] text-faint">
          {mode === 'all'
            ? 'All enabled tools are described to the model.'
            : `On demand: the model loads tools as it needs them (≈${formatCompact(tokensOf(loaded))} of ${formatCompact(settings.toolBudgetTokens)} tokens used).`}
        </div>
      )}
      {mode === 'dynamic' && (
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-[11px] text-muted">
            <span>Loaded in this chat</span>
            {loaded.length > 0 && (
              <button className="hover:text-accent" onClick={() => mutate((c) => ({ ...c, activeTools: [] }))}>
                Clear
              </button>
            )}
          </div>
          {loaded.length === 0 ? (
            <div className="text-[11px] text-faint">None yet.</div>
          ) : (
            <div className="flex flex-wrap gap-1">
              {loaded.map((n) => (
                <button
                  key={n}
                  title="Unload"
                  onClick={() => mutate((c) => ({ ...c, activeTools: (c.activeTools ?? []).filter((x) => x !== n) }))}
                  className="rounded border border-border bg-panel-2 px-1.5 py-0.5 font-mono text-[11px] hover:border-danger/50 hover:text-danger"
                >
                  {findTool(n)?.name ?? n} ✕
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </Section>
  )
}

export function InferencePanel() {
  // Only the fields shown here: `current` itself is a new object on every streaming flush, which
  // would re-render all the sliders (and the tool list) about 11 times a second.
  const current = useChat(
    useShallow((s) => (s.current ? { sampling: s.current.sampling, presetId: s.current.presetId, systemPrompt: s.current.systemPrompt, modelId: s.current.modelId } : null))
  )
  const presets = useChat((s) => s.presets)
  const mutate = useChat((s) => s.mutateCurrent)
  const { applyPreset, savePreset, deletePreset } = useChat.getState()
  const settings = useApp((s) => s.settings)
  const updateSettings = useApp((s) => s.updateSettings)
  const [saveOpen, setSaveOpen] = useState(false)
  const [presetName, setPresetName] = useState('')
  const [varietyOpen, setVarietyOpen] = useState(false)
  // An OpenRouter model honours only some settings; the rest are greyed. Undefined for local and rented models.
  // The entry is picked from the store and the rest worked out here: a selector that makes a new object each time renders forever.
  // With no active model, the chat's own one is connected by the next message.
  const activeEntry = useModels((st) => st.models.find((m) => m.id === (st.activeModelId ?? current?.modelId)))
  const supports = supportsOf(activeEntry)

  if (!current || !settings) {
    return <aside className="w-80 shrink-0 border-l border-border bg-panel p-4 text-xs text-faint">Open a chat to edit its settings.</aside>
  }

  const s = current.sampling
  const setS = <K extends keyof SamplingParams>(key: K, value: SamplingParams[K]): void =>
    mutate((c) => ({ ...c, sampling: { ...c.sampling, [key]: value } }))
  const activePreset = presets.find((p) => p.id === current.presetId)
  const varietyOn = (s.dryMultiplier > 0 ? 1 : 0) + (s.xtcProbability > 0 ? 1 : 0) + (s.avoidWords.length > 0 ? 1 : 0)

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
          <IconButton
            size="sm"
            label="Reset to defaults"
            onClick={() =>
              mutate((c) => ({
                ...c,
                sampling: modeSampling(c.re ? 're' : 'chat', modeDefaultsOf(useModels.getState().activeModelId ?? c.modelId), settings.defaultSampling)
              }))
            }
          >
            <RotateCcw size={13} />
          </IconButton>
        }
      >
        <SamplingFields sampling={s} onChange={setS} supports={supports} />
      </Section>

      <section className="border-b border-border px-4 py-4">
        <Collapsible.Root open={varietyOpen} onOpenChange={setVarietyOpen}>
          <Collapsible.Trigger className="flex w-full items-center gap-1 text-[11px] font-semibold tracking-wide text-faint uppercase hover:text-text">
            <ChevronRight size={14} className={varietyOpen ? 'rotate-90 transition-transform' : 'transition-transform'} />
            Repetition and variety
            {varietyOn > 0 && <span className="font-normal normal-case"> · {varietyOn} on</span>}
          </Collapsible.Trigger>
          <Collapsible.Content className="mt-3 space-y-3">
            <VarietyFields sampling={s} onChange={setS} supports={supports} />
          </Collapsible.Content>
        </Collapsible.Root>
      </section>

      <Section title="Thinking">
        <ThinkingFields sampling={s} onChange={setS} supports={supports} />
      </Section>

      <Section title="Adapters">
        <AdapterPanel />
      </Section>

      <ToolsSection />

      <Section title="Structured output">
        <Field inline label="Constrain to a JSON schema" hint={supports && !supports.jsonSchema ? NOT_SUPPORTED : undefined}>
          <Switch
            disabled={!!supports && !supports.jsonSchema}
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
