import { ChevronRight } from 'lucide-react'
import { useState } from 'react'
import { adoptSampling, MODE_LABELS, SHARED_SAMPLING_KEYS, type AppMode, type LeftoverSampling } from '@shared/modes'
import type { SamplingParams } from '@shared/types'
import { cn } from '../../lib/format'
import { SamplingFields, ThinkingFields, VarietyFields } from '../chat/InferencePanel'
import { Button, Field, Switch } from '../ui'

export const FALLBACK: Record<AppMode, string> = {
  chat: 'Uses the new-chat sampling in Settings.',
  re: 'Uses the built-in values for tool use.',
  adventure: "Uses the app's built-in sampling.",
  writing: 'Uses the fiction sampling new projects start with.',
  coding: "Uses RulZ Coding Agent's own recommended values. Its own settings win where it has them."
}

const LEFTOVER_LABELS: Record<(typeof SHARED_SAMPLING_KEYS)[number], string> = {
  temperature: 'temperature',
  topK: 'top K',
  topP: 'top P',
  minP: 'min P',
  repeatPenalty: 'repeat penalty',
  presencePenalty: 'presence penalty',
  dryMultiplier: 'DRY multiplier',
  dryBase: 'DRY base',
  dryAllowedLength: 'DRY allowed length',
  dryPenaltyLastN: 'DRY last N',
  drySequenceBreakers: 'DRY sequence breakers',
  xtcProbability: 'XTC probability',
  xtcThreshold: 'XTC threshold'
}

/** The leftover values as a short list: "temperature 1, top K 0". At most six, then "and 2 more". */
export function leftoverText(leftover: LeftoverSampling): string {
  const parts = SHARED_SAMPLING_KEYS.filter((k) => leftover[k] !== undefined).map((k) => {
    const v = leftover[k]
    return `${LEFTOVER_LABELS[k]} ${Array.isArray(v) ? `${v.length} item${v.length === 1 ? '' : 's'}` : String(v)}`
  })
  return parts.length > 6 ? `${parts.slice(0, 6).join(', ')} and ${parts.length - 6} more` : parts.join(', ')
}

/**
 * One mode's own sampling: the switch and, when on, the sampling fields. Shared by the mode editor and the model picker.
 * `onChange` gets the whole next state (the switch and the values); the host decides when it is saved.
 */
export function OwnSampling({
  mode,
  on,
  sampling,
  appSampling,
  leftover,
  note,
  onChange
}: {
  mode: AppMode
  on: boolean
  sampling: SamplingParams | null
  /** The app's new-chat sampling; the switch needs it to start from the mode's built-in values. */
  appSampling: SamplingParams | undefined
  /** API-server sampling values found in the mode's stored load settings, offered for adoption; null = none. */
  leftover: LeftoverSampling | null
  /** An extra sentence under the fields, e.g. when the values are saved. */
  note?: string
  onChange: (next: { on: boolean; sampling: SamplingParams | null }) => void
}) {
  const [varietyOpen, setVarietyOpen] = useState(false)
  const honoursLimits = mode === 'chat' || mode === 're'
  // Coding takes the thinking fields, but the Coding Agent keeps its own reply length and seed.
  const honoursThinking = honoursLimits || mode === 'coding'
  const editSampling = <K extends keyof SamplingParams>(key: K, value: SamplingParams[K]): void => {
    if (sampling) onChange({ on, sampling: { ...sampling, [key]: value } })
  }

  return (
    <section className="space-y-3" aria-label="Own sampling">
      <Field inline label="Own sampling" hint={mode === 'coding' ? "What RulZ Coding Agent's normal tasks use; its own values win there" : `Only for ${MODE_LABELS[mode]}`}>
        <Switch
          checked={on}
          onCheckedChange={(next) => onChange({ on: next, sampling: next ? (sampling ?? (appSampling ? structuredClone(adoptSampling(mode, appSampling, {})) : null)) : sampling })}
          disabled={!appSampling}
        />
      </Field>
      {on && sampling ? (
        <div className="space-y-3">
          <SamplingFields sampling={sampling} onChange={editSampling} limits={honoursLimits} />
          <button
            type="button"
            aria-expanded={varietyOpen}
            onClick={() => setVarietyOpen(!varietyOpen)}
            className="flex w-full items-center gap-1 text-[11px] font-semibold tracking-wide text-faint uppercase hover:text-text"
          >
            <ChevronRight size={14} className={cn('transition-transform', varietyOpen && 'rotate-90')} />
            Repetition and variety
          </button>
          {varietyOpen && <VarietyFields sampling={sampling} onChange={editSampling} />}
          {honoursThinking && <ThinkingFields sampling={sampling} onChange={editSampling} />}
        </div>
      ) : (
        <>
          <p className="text-xs text-muted">{FALLBACK[mode]}</p>
          {leftover && appSampling && (
            <div className="space-y-1.5">
              <p className="text-[11.5px] text-muted">
                These load settings hold sampling values that only the API server used: {leftoverText(leftover)}.
              </p>
              <Button size="sm" onClick={() => onChange({ on: true, sampling: adoptSampling(mode, appSampling, leftover) })}>
                Use these for {MODE_LABELS[mode]}
              </Button>
            </div>
          )}
        </>
      )}
      {note && <p className="text-[11.5px] text-muted">{note}</p>}
    </section>
  )
}
