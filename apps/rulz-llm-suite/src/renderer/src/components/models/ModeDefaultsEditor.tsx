import { RotateCcw } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { toast } from 'sonner'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import { APP_MODES, leftoverSampling, MODE_LABELS, modeDefaultsEmpty, type AppMode, type ModeDefaults } from '@shared/modes'
import type { LoadConfig, ModelEntry, ModelSampling, SamplingParams } from '@shared/types'
import { errorMessage, invoke } from '../../lib/api'
import { modesFor } from '../../lib/modes'
import { useApp } from '../../stores/app'
import { useModels } from '../../stores/models'
import { Button, Field, Select, Switch, Textarea } from '../ui'
import { EstimateView, useEstimate } from './EstimateView'
import { LoadSettingsForm } from './LoadSettingsForm'
import { OwnSampling } from './OwnSampling'

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

/**
 * What a draft saves. The mode's load keeps the API-server sampling it was stored with (the form does not edit it); saved
 * with own sampling on, it gets the model's default one (`baseSampling`), so leftover values are not offered again.
 */
const valueOf = (d: Draft, baseSampling?: ModelSampling): ModeDefaults | null => {
  const sampling = d.ownSampling && d.sampling ? { ...d.sampling, jsonSchema: '' } : null
  const load = d.ownLoad && d.load ? (sampling && baseSampling ? { ...d.load, sampling: baseSampling } : d.load) : null
  const value: ModeDefaults = { load, sampling, systemPrompt: d.prompt }
  return modeDefaultsEmpty(value) ? null : value
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
    const value = draft ? valueOf(draft, base?.sampling) : null
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
                <LoadSettingsForm model={model} config={view.load} onChange={(c) => edit({ load: c })} estimate={estimate} remote={remote} hideApiSampling />
                <EstimateView estimate={estimate} />
              </>
            ) : (
              <p className="text-xs text-muted">Uses the model&apos;s default load settings.</p>
            )}
          </section>

          <OwnSampling
            mode={mode}
            on={view.ownSampling}
            sampling={view.sampling}
            appSampling={appSampling}
            leftover={base ? leftoverSampling(saved?.load, base) : null}
            onChange={(n) => edit({ ownSampling: n.on, sampling: n.sampling })}
          />

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
