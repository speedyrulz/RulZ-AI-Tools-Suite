import { ChevronRight, RotateCcw } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { toast } from 'sonner'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import { APP_MODES, builtInSampling, MODE_LABELS, modeDefaultsEmpty, type AppMode, type ModeDefaults } from '@shared/modes'
import type { LoadConfig, ModelEntry, SamplingParams } from '@shared/types'
import { errorMessage, invoke } from '../../lib/api'
import { cn } from '../../lib/format'
import { modesFor } from '../../lib/modes'
import { useApp } from '../../stores/app'
import { useModels } from '../../stores/models'
import { SamplingFields, ThinkingFields, VarietyFields } from '../chat/InferencePanel'
import { Button, Field, Select, Switch, Textarea } from '../ui'
import { EstimateView, useEstimate } from './EstimateView'
import { LoadSettingsForm } from './LoadSettingsForm'

/** 'all' = the model's default load settings (the host shows its own view for it). */
export type ModeScope = 'all' | AppMode

/** What the editor holds for one mode while it is being edited. */
interface Draft {
  ownLoad: boolean
  load: LoadConfig | null
  ownSampling: boolean
  sampling: SamplingParams | null
  prompt: string
}

const draftOf = (saved: ModeDefaults | undefined): Draft => ({
  ownLoad: !!saved?.load,
  load: saved?.load ?? null,
  ownSampling: !!saved?.sampling,
  sampling: saved?.sampling ?? null,
  prompt: saved?.systemPrompt ?? ''
})

const valueOf = (d: Draft): ModeDefaults | null => {
  const value: ModeDefaults = {
    load: d.ownLoad ? d.load : null,
    sampling: d.ownSampling && d.sampling ? { ...d.sampling, jsonSchema: '' } : null,
    systemPrompt: d.prompt
  }
  return modeDefaultsEmpty(value) ? null : value
}

const FALLBACK: Record<AppMode, string> = {
  chat: 'Uses the new-chat sampling in Settings.',
  re: 'Uses the built-in values for tool use.',
  adventure: "Uses the app's built-in sampling.",
  writing: 'Uses the fiction sampling new projects start with.'
}

const WHEN: Record<AppMode, string> = {
  chat: 'Used for new chats; existing ones keep their settings.',
  re: 'Used for new sessions; existing ones keep their settings.',
  adventure: 'Used from the next passage. A temperature set on the adventure wins.',
  writing: 'New projects start with this sampling; the prompt is used for every request.'
}

/**
 * "Settings for" a model: all modes (the host's own default load settings view, passed as `allView`) or one mode with its
 * own load settings, sampling and system prompt. Unsaved changes are kept per mode while the Select moves around.
 */
export function ModeDefaultsEditor(props: {
  modelId: string
  model: ModelEntry | null
  remote?: boolean
  /** The rental whose GPUs the memory estimate is for (remote models). */
  rentalId?: number | null
  scope: ModeScope
  onScope: (s: ModeScope) => void
  allView: ReactNode
}) {
  // A new model starts with no drafts.
  return <Editor key={props.modelId} {...props} />
}

function Editor({ modelId, model, remote, rentalId, scope, onScope, allView }: Parameters<typeof ModeDefaultsEditor>[0]) {
  const all = useModels((s) => s.modeDefaults)
  const settings = useApp((s) => s.settings)
  const modes = modesFor(all, modelId)
  const [drafts, setDrafts] = useState<Partial<Record<AppMode, Draft>>>({})
  const [base, setBase] = useState<LoadConfig | null>(null)
  const [busy, setBusy] = useState(false)
  const [varietyOpen, setVarietyOpen] = useState(false)
  const baseSeq = useRef(0)

  // The model's default load settings, as they are now: a mode's own settings start from them.
  useEffect(() => {
    const seq = ++baseSeq.current
    invoke('models:getDefaults', modelId)
      .then((d) => seq === baseSeq.current && setBase(d.config))
      .catch(() => seq === baseSeq.current && setBase(useApp.getState().settings?.defaultLoadConfig ?? DEFAULT_LOAD_CONFIG))
  }, [modelId, scope])

  const mode: AppMode | null = scope === 'all' ? null : scope
  const saved = mode ? modes?.[mode] : undefined
  const view = mode ? (drafts[mode] ?? draftOf(saved)) : null
  const appSampling = settings?.defaultSampling
  const estimate = useEstimate(modelId, view?.load ?? base ?? DEFAULT_LOAD_CONFIG, rentalId)

  const edit = (patch: Partial<Draft>): void => {
    if (!mode) return
    setDrafts((d) => ({ ...d, [mode]: { ...(d[mode] ?? draftOf(saved)), ...patch } }))
  }
  const editSampling = <K extends keyof SamplingParams>(key: K, value: SamplingParams[K]): void => {
    if (view?.sampling) edit({ sampling: { ...view.sampling, [key]: value } })
  }

  const finish = async (m: AppMode, draft: Draft | undefined): Promise<void> => {
    // Read the saved state again, then drop the draft unless it was edited meanwhile.
    await useModels.getState().refreshModeDefaults()
    setDrafts((cur) => {
      if (cur[m] !== draft) return cur
      return omit(cur, m)
    })
  }

  const save = async (): Promise<void> => {
    if (!mode || !drafts[mode]) return
    const m = mode
    const draft = drafts[m]
    const value = draft ? valueOf(draft) : null
    setBusy(true)
    try {
      await invoke('models:setModeDefaults', modelId, m, value)
      toast.success(value ? `Saved the ${MODE_LABELS[m]} settings` : `Cleared the ${MODE_LABELS[m]} settings`)
      await finish(m, draft)
    } catch (e) {
      toast.error(`Could not save the ${MODE_LABELS[m]} settings: ${errorMessage(e)}`)
    } finally {
      setBusy(false)
    }
  }

  const reset = async (): Promise<void> => {
    if (!mode) return
    const m = mode
    const draft = drafts[m]
    setBusy(true)
    try {
      if (saved) {
        await invoke('models:setModeDefaults', modelId, m, null)
        toast.success(`${MODE_LABELS[m]} uses the default settings again`)
      }
      await finish(m, draft)
    } catch (e) {
      toast.error(`Could not reset the ${MODE_LABELS[m]} settings: ${errorMessage(e)}`)
    } finally {
      setBusy(false)
    }
  }

  const honoursLimits = mode === 'chat' || mode === 're'

  return (
    <div className="space-y-4">
      <Field label="Settings for">
        <Select value={scope} onChange={(e) => onScope(e.target.value as ModeScope)} aria-label="Settings for">
          <option value="all">All modes (default)</option>
          {APP_MODES.map((m) => (
            <option key={m} value={m}>
              {MODE_LABELS[m]}
              {modeDefaultsEmpty(modes?.[m]) ? '' : ' · own settings'}
            </option>
          ))}
        </Select>
      </Field>

      {!mode || !view ? (
        allView
      ) : (
        <div className="space-y-5">
          <p className="text-xs text-muted">
            {WHEN[mode]} Load settings apply when the model is loaded.
          </p>

          <section className="space-y-3" aria-label="Own load settings">
            <Field inline label="Own load settings" hint={`Only for ${MODE_LABELS[mode]}`}>
              <Switch
                checked={view.ownLoad}
                disabled={!base && !view.load}
                onCheckedChange={(on) => edit({ ownLoad: on, load: on ? (view.load ?? base ?? settings?.defaultLoadConfig ?? DEFAULT_LOAD_CONFIG) : view.load })}
              />
            </Field>
            {view.ownLoad && view.load ? (
              <>
                <LoadSettingsForm model={model} config={view.load} onChange={(c) => edit({ load: c })} estimate={estimate} remote={remote} />
                <EstimateView estimate={estimate} />
              </>
            ) : (
              <p className="text-xs text-muted">Uses the model&apos;s default load settings.</p>
            )}
          </section>

          <section className="space-y-3" aria-label="Own sampling">
            <Field inline label="Own sampling" hint={`Only for ${MODE_LABELS[mode]}`}>
              <Switch
                checked={view.ownSampling}
                onCheckedChange={(on) =>
                  edit({
                    ownSampling: on,
                    sampling: on ? (view.sampling ?? (appSampling ? structuredClone(builtInSampling(mode, appSampling)) : null)) : view.sampling
                  })
                }
                disabled={!appSampling}
              />
            </Field>
            {view.ownSampling && view.sampling ? (
              <div className="space-y-3">
                <SamplingFields sampling={view.sampling} onChange={editSampling} limits={honoursLimits} />
                <button
                  type="button"
                  aria-expanded={varietyOpen}
                  onClick={() => setVarietyOpen(!varietyOpen)}
                  className="flex w-full items-center gap-1 text-[11px] font-semibold tracking-wide text-faint uppercase hover:text-text"
                >
                  <ChevronRight size={14} className={cn('transition-transform', varietyOpen && 'rotate-90')} />
                  Repetition and variety
                </button>
                {varietyOpen && <VarietyFields sampling={view.sampling} onChange={editSampling} />}
                {honoursLimits && <ThinkingFields sampling={view.sampling} onChange={editSampling} />}
              </div>
            ) : (
              <p className="text-xs text-muted">{FALLBACK[mode]}</p>
            )}
          </section>

          <section className="space-y-2" aria-label="System prompt">
            <Field
              label="System prompt"
              hint={mode === 'chat' ? "Used for new chats with this model. Empty = the app's default system prompt." : 'Added after the built-in prompt. Empty = nothing is added.'}
            >
              <Textarea
                rows={5}
                className="max-h-[40vh] min-h-20"
                aria-label={`System prompt for ${MODE_LABELS[mode]}`}
                value={view.prompt}
                onChange={(e) => edit({ prompt: e.target.value })}
              />
            </Field>
          </section>

          <div className="flex items-center justify-end gap-2">
            <Button variant="ghost" disabled={busy || (!saved && !drafts[mode])} onClick={() => void reset()}>
              <RotateCcw size={14} /> Reset
            </Button>
            <Button variant="primary" disabled={busy || !drafts[mode]} onClick={() => void save()}>
              Save {MODE_LABELS[mode]} settings
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

function omit<T extends object>(o: T, k: keyof T): T {
  const { [k]: _gone, ...rest } = o
  return rest as T
}
