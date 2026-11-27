import type { LoraMergeDtype, LoraMergeMode } from '@shared/loraMerge'
import { DTYPE_CHOICES, LORA_RANK_CHOICES, MODE_TEXT, NAME_MAX, validLoraOutputName } from '../../lib/loraMerge'
import { useLoraMerge } from '../../stores/loraMerge'
import { Card, DebouncedInput, Field, Segmented } from '../ui'

const SELECT_CLASS =
  'h-9 w-full rounded-lg border border-border bg-panel px-3 text-[13px] focus:border-accent focus:outline-none disabled:cursor-not-allowed disabled:opacity-45'

export function ResultCard() {
  const mode = useLoraMerge((s) => s.mode)
  const rank = useLoraMerge((s) => s.rank)
  const dtype = useLoraMerge((s) => s.dtype)
  const outputName = useLoraMerge((s) => s.outputName)
  const setOptions = useLoraMerge((s) => s.setOptions)
  const setOutputName = useLoraMerge((s) => s.setOutputName)
  const nameBad = !validLoraOutputName(outputName)

  return (
    <Card title="Result" description="How the merged LoRA is built and saved.">
      <div className="space-y-5">
        <Field label="Rank" hint={MODE_TEXT[mode].text}>
          <Segmented
            label="Rank mode"
            value={mode}
            onChange={(m: LoraMergeMode) => setOptions({ mode: m })}
            options={[
              { value: 'concat', label: MODE_TEXT.concat.label },
              { value: 'shrink', label: MODE_TEXT.shrink.label }
            ]}
          />
        </Field>
        {mode === 'shrink' && (
          <div className="max-w-xs">
            <Field label="Target rank" hint="A layer keeps at most this many ranks.">
              <select aria-label="Target rank" className={SELECT_CLASS} value={rank} onChange={(e) => setOptions({ rank: Number(e.target.value) })}>
                {LORA_RANK_CHOICES.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
            </Field>
          </div>
        )}
        <div className="max-w-xs">
          <Field label="Precision" hint="The number format of the saved file.">
            <select aria-label="Precision" className={SELECT_CLASS} value={dtype} onChange={(e) => setOptions({ dtype: e.target.value as LoraMergeDtype })}>
              {DTYPE_CHOICES.map((d) => (
                <option key={d.value} value={d.value}>
                  {d.label}
                </option>
              ))}
            </select>
          </Field>
        </div>
        <div className="max-w-md">
          <Field
            label="Output name"
            hint={
              nameBad ? (
                <span className="text-danger-text">Use letters, digits, spaces and - _ . only, at most {NAME_MAX} characters.</span>
              ) : (
                <>Saved as {outputName.trim()}.safetensors in models/loras/merged/.</>
              )
            }
          >
            <DebouncedInput aria-label="Output name" value={outputName} delay={250} onCommit={setOutputName} />
          </Field>
        </div>
      </div>
    </Card>
  )
}
