import { ChevronDown, FolderOpen } from 'lucide-react'
import { useState } from 'react'
import type { Theme } from '@shared/types'
import { invoke } from '../lib/api'
import { cn } from '../lib/format'
import { activeModelId, useApp } from '../stores/app'
import { ApprovalChooser, ModelSelect } from './common'
import { Button, Dialog, Field, Input, NumberInput } from './ui'

const THEMES: { value: Theme; label: string }[] = [
  { value: 'dark', label: 'Dark' },
  { value: 'light', label: 'Light' },
  { value: 'system', label: 'Same as Windows' }
]

export function SettingsDialog() {
  const open = useApp((s) => s.settingsOpen)
  const setOpen = useApp((s) => s.setSettingsOpen)
  const settings = useApp((s) => s.settings)
  const connection = useApp((s) => s.connection)
  const update = useApp((s) => s.updateSettings)
  const setTheme = useApp((s) => s.setTheme)
  const [details, setDetails] = useState(false)
  if (!settings) return null

  return (
    <Dialog open={open} onOpenChange={setOpen} title="Settings" className="w-[min(620px,calc(100vw-32px))]">
      <div className="space-y-6 px-5 py-5">
        <Field label="AI model" hint="Models marked “ready” start right away">
          {connection.models.length ? (
            <ModelSelect models={connection.models} value={activeModelId(settings, connection)} onChange={(model) => void update({ model })} />
          ) : (
            <p className="text-[13px] text-muted">Connect to RulZ LLM Suite to see its models.</p>
          )}
        </Field>

        <Field label="When should I ask you first?">
          <ApprovalChooser value={settings.approval} onChange={(approval) => void update({ approval })} />
        </Field>

        <Field label="Look">
          <div className="flex gap-1.5">
            {THEMES.map((t) => (
              <Button key={t.value} variant={settings.theme === t.value ? 'primary' : 'secondary'} onClick={() => void setTheme(t.value)}>
                {t.label}
              </Button>
            ))}
          </div>
        </Field>

        <div>
          <button className="flex items-center gap-1 text-[13px] text-muted hover:text-text" onClick={() => setDetails(!details)}>
            <ChevronDown size={14} className={cn('transition-transform', !details && '-rotate-90')} />
            Show details
          </button>
          {details && (
            <div className="mt-3 space-y-4 rounded-lg border border-border bg-panel-2/50 p-4">
              <Field label="RulZ LLM Suite address" hint={connection.status === 'connected' ? `Connected to ${connection.url}` : 'Not connected'}>
                <Input
                  defaultValue={settings.serverUrl}
                  placeholder="Find it automatically (ports 1234 and 1235)"
                  onBlur={(e) => e.target.value.trim() !== settings.serverUrl && void update({ serverUrl: e.target.value.trim() })}
                />
              </Field>
              <Field label="API key" hint="Only if you set one in RulZ LLM Suite">
                <Input
                  type="password"
                  defaultValue={settings.apiKey}
                  placeholder="None"
                  onBlur={(e) => e.target.value !== settings.apiKey && void update({ apiKey: e.target.value })}
                />
              </Field>
              <Field label="New projects go in">
                <div className="flex gap-2">
                  <Input value={settings.projectsDir} readOnly />
                  <Button
                    onClick={async () => {
                      const dir = await invoke('dialog:pickFolder', { title: 'Where should new projects go?', defaultPath: settings.projectsDir })
                      if (dir) void update({ projectsDir: dir })
                    }}
                  >
                    <FolderOpen size={14} /> Choose
                  </Button>
                </div>
              </Field>
              <Field label="Most steps per task" hint="I pause and ask after this many" inline>
                <NumberInput value={settings.maxSteps} min={5} max={500} onChange={(v) => v !== null && v >= 5 && v <= 500 && void update({ maxSteps: Math.round(v) })} />
              </Field>
              <Field label="Time limit for one command (minutes)" inline>
                <NumberInput value={settings.commandTimeoutMin} min={1} max={120} onChange={(v) => v && void update({ commandTimeoutMin: v })} />
              </Field>
            </div>
          )}
        </div>
      </div>
    </Dialog>
  )
}
