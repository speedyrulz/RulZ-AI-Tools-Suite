import type { CloudState } from '@shared/cloud'
import { isChannelBusy, useCloud } from '../../stores/cloud'
import { Field, Switch } from '../ui'

export function PrivacyCard({ state }: { state: CloudState }) {
  const run = useCloud((s) => s.run)
  const busy = useCloud((s) => isChannelBusy(s, 'cloud:setOptions'))
  return (
    <section className="rounded-xl border border-border bg-panel p-5" aria-label="Privacy">
      <h2 className="font-[family-name:var(--font-display)] text-sm font-semibold">Privacy</h2>
      <div className="mt-3">
        <Field
          label="Allow providers that may keep or train on prompts"
          hint="Off: requests only go to providers that say they do not keep or train on them. Some free models only work with this on."
          inline
        >
          <Switch checked={state.options.allowTraining} disabled={busy} onCheckedChange={(on) => void run('cloud:setOptions', { allowTraining: on })} />
        </Field>
      </div>
    </section>
  )
}
