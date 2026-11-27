import type { GpuInfo, OwnServerSettings, RunningServer } from '@shared/types'
import { DebouncedInput, Card, Field, NumberInput, Switch } from '../ui'
import { DevicePicker } from './DevicePicker'

export function OwnServerCard({
  own,
  gpus,
  servers,
  locked,
  onChange
}: {
  own: OwnServerSettings
  gpus: GpuInfo[]
  servers: RunningServer[]
  /** The server is running or changing state: settings apply from the next start. */
  locked: boolean
  onChange: (patch: Partial<OwnServerSettings>) => void
}) {
  return (
    <Card
      title="Own server"
      description={locked ? 'Stop the server to change these settings.' : 'The app starts ComfyUI with these settings.'}
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Field label="WSL distro">
          <DebouncedInput value={own.distro} onCommit={(v) => onChange({ distro: v })} disabled={locked} />
        </Field>
        <Field label="ComfyUI folder" hint="A path inside WSL. ~ means your home folder there.">
          <DebouncedInput mono value={own.comfyDir} onCommit={(v) => onChange({ comfyDir: v })} disabled={locked} />
        </Field>
        <Field label="Python" hint="Relative to the ComfyUI folder, or an absolute path.">
          <DebouncedInput mono value={own.python} onCommit={(v) => onChange({ python: v })} disabled={locked} />
        </Field>
        <Field label="Base folder" hint="Where the own server keeps its output, input, user and custom_nodes folders.">
          <DebouncedInput mono value={own.baseDir} onCommit={(v) => onChange({ baseDir: v })} disabled={locked} />
        </Field>
        <Field label="Models folder" hint="Leave empty to use the models folder inside the ComfyUI folder.">
          <DebouncedInput
            mono
            value={own.modelsDir}
            placeholder="<ComfyUI folder>/models"
            onCommit={(v) => onChange({ modelsDir: v })}
            disabled={locked}
          />
        </Field>
        <Field label="First port" hint="The next free port at or above this one is used.">
          <NumberInput value={own.portStart} min={1024} max={65535} onCommit={(v) => onChange({ portStart: v })} disabled={locked} />
        </Field>
        <Field label="Extra options" hint="Added to the ComfyUI command line, split on spaces.">
          <DebouncedInput mono value={own.extraArgs} onCommit={(v) => onChange({ extraArgs: v })} disabled={locked} />
        </Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="Minimum free memory (GiB)" hint="Below this you must confirm before starting.">
            <NumberInput value={own.minFreeRamGiB} min={0} max={1024} onCommit={(v) => onChange({ minFreeRamGiB: v })} disabled={locked} />
          </Field>
          <Field label="Stop wait (seconds)" hint="Time to wait before the server is forced to stop.">
            <NumberInput value={own.stopTimeoutSec} min={1} max={600} onCommit={(v) => onChange({ stopTimeoutSec: v })} disabled={locked} />
          </Field>
        </div>
      </div>

      <div className="mt-5 flex items-start justify-between gap-4 rounded-lg border border-border bg-panel-2 px-4 py-3">
        <div>
          <label htmlFor="load-custom-nodes" className="text-[13px] font-medium">
            Load my custom nodes
          </label>
          <p className="text-xs text-faint">
            Needed for models whose loaders come from node packs. Off by default: the own server then loads no node packs.
          </p>
        </div>
        <Switch id="load-custom-nodes" checked={own.loadCustomNodes} onCheckedChange={(v) => onChange({ loadCustomNodes: v })} disabled={locked} />
      </div>

      <h3 className="mt-5 mb-2 text-[13px] font-medium">Device</h3>
      <DevicePicker gpus={gpus} device={own.device} servers={servers} disabled={locked} onChange={(d) => onChange({ device: d })} />
    </Card>
  )
}
