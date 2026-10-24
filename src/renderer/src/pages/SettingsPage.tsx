import { CircleAlert, CircleCheck, FolderOpen, FolderSearch, RefreshCw } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import type { LoadConfig, SamplingParams } from '@shared/types'
import { LoadSettingsForm } from '../components/models/LoadSettingsForm'
import { Button, Field, Input, Kbd, Select, Switch, Textarea } from '../components/ui'
import { SHORTCUTS } from '../hooks/useShortcuts'
import { invoke } from '../lib/api'
import { useApp } from '../stores/app'

function Card({ title, description, children }: { title: string; description?: string; children: ReactNode }) {
  return (
    <section className="rounded-xl border border-border bg-panel p-5">
      <h2 className="text-sm font-semibold">{title}</h2>
      {description && <p className="mt-0.5 text-xs text-muted">{description}</p>}
      <div className="mt-4 space-y-4">{children}</div>
    </section>
  )
}

/** Text input that saves on blur/Enter instead of on every keystroke. */
function PathInput({ value, placeholder, onCommit }: { value: string; placeholder?: string; onCommit: (v: string) => void }) {
  const [draft, setDraft] = useState(value)
  useEffect(() => {
    setDraft(value)
  }, [value])
  return (
    <Input
      className="font-mono text-xs"
      value={draft}
      placeholder={placeholder}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={() => draft !== value && onCommit(draft.trim())}
      onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
    />
  )
}

export function SettingsPage() {
  const settings = useApp((s) => s.settings)
  const engine = useApp((s) => s.engine)
  const checking = useApp((s) => s.engineChecking)
  const info = useApp((s) => s.info)
  const update = useApp((s) => s.updateSettings)
  const refreshEngine = useApp((s) => s.refreshEngine)
  const [loadDefaults, setLoadDefaults] = useState<LoadConfig | null>(settings?.defaultLoadConfig ?? null)

  if (!settings) return null

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-3xl space-y-5 px-6 py-8">
        <h1 className="text-lg font-semibold">Settings</h1>

        <Card title="Engine" description="RulZ LLM Suite runs models with your own llama.cpp build (llama-server).">
          <Field label="llama.cpp folder" hint="Folder containing llama-server.exe. Empty = auto-detect.">
            <div className="flex gap-2">
              <PathInput
                value={settings.enginePath}
                placeholder={engine?.exePath ?? 'C:\\coding\\llama.cpp\\build-cuda\\bin\\Release'}
                onCommit={(v) => void update({ enginePath: v })}
              />
              <Button
                onClick={async () => {
                  const dir = await invoke('dialog:pickFolder', { title: 'Folder containing llama-server.exe' })
                  if (dir) await update({ enginePath: dir })
                }}
              >
                <FolderSearch size={14} /> Browse
              </Button>
              <Button onClick={() => void refreshEngine()} disabled={checking}>
                <RefreshCw size={14} className={checking ? 'animate-spin' : undefined} /> Test
              </Button>
            </div>
          </Field>
          {engine && (
            <div className={`rounded-lg border p-3 text-xs ${engine.ok ? 'border-ok/30 bg-ok/5' : 'border-danger/40 bg-danger/5'}`}>
              <div className="flex items-center gap-2 font-medium">
                {engine.ok ? <CircleCheck size={14} className="text-ok" /> : <CircleAlert size={14} className="text-danger" />}
                {engine.ok ? `llama-server ${engine.version ?? ''}` : 'llama-server not available'}
              </div>
              {engine.exePath && <div className="mt-1 font-mono text-[11px] break-all text-muted">{engine.exePath}</div>}
              {engine.error && <pre className="mt-2 font-mono text-[11px] whitespace-pre-wrap text-danger">{engine.error}</pre>}
              {engine.ok && (
                <div className="mt-2 space-y-0.5">
                  {engine.devices.length === 0 && <div className="text-warn">No GPU devices reported — models will run on the CPU.</div>}
                  {engine.devices.map((d, i) => (
                    <div key={d.id} className="text-muted">
                      GPU{i} ({d.id}): {d.name} — {d.freeMiB?.toLocaleString()} / {d.totalMiB?.toLocaleString()} MiB free
                    </div>
                  ))}
                </div>
              )}
              {!engine.ok && (
                <div className="mt-2 text-muted">
                  Looked in: <span className="font-mono text-[11px]">{engine.searched.join(' · ')}</span>
                </div>
              )}
            </div>
          )}
        </Card>

        <Card title="Models" description="Folder scanned (recursively) for .gguf files. Changes are picked up automatically.">
          <Field label="Models folder">
            <div className="flex gap-2">
              <PathInput value={settings.modelsDir} onCommit={(v) => v && void update({ modelsDir: v })} />
              <Button
                onClick={async () => {
                  const dir = await invoke('dialog:pickFolder', { title: 'Choose your models folder', defaultPath: settings.modelsDir })
                  if (dir) await update({ modelsDir: dir })
                }}
              >
                <FolderSearch size={14} /> Browse
              </Button>
            </div>
          </Field>
          <Field inline label="Unload the previous model when loading another" hint="Turn off to keep several models loaded at once">
            <Switch checked={settings.unloadPreviousOnLoad} onCheckedChange={(v) => void update({ unloadPreviousOnLoad: v })} />
          </Field>
        </Card>

        <Card title="Default load settings" description="Used for models without their own saved settings.">
          {loadDefaults && <LoadSettingsForm model={null} config={loadDefaults} onChange={setLoadDefaults} />}
          <div className="flex justify-end">
            <Button variant="primary" onClick={() => loadDefaults && void update({ defaultLoadConfig: loadDefaults })}>
              Save load defaults
            </Button>
          </div>
        </Card>

        <Card title="New chats" description="Starting values for new conversations.">
          <Field label="Default system prompt">
            <Textarea
              rows={4}
              defaultValue={settings.defaultSystemPrompt}
              onBlur={(e) => e.target.value !== settings.defaultSystemPrompt && void update({ defaultSystemPrompt: e.target.value })}
            />
          </Field>
          <div className="grid grid-cols-2 gap-4">
            <NumberSetting label="Temperature" value={settings.defaultSampling.temperature} step={0.05} onChange={(v) => void update({ defaultSampling: withSampling(settings.defaultSampling, 'temperature', v) })} />
            <NumberSetting label="Top P" value={settings.defaultSampling.topP} step={0.01} onChange={(v) => void update({ defaultSampling: withSampling(settings.defaultSampling, 'topP', v) })} />
            <NumberSetting label="Top K" value={settings.defaultSampling.topK} step={1} onChange={(v) => void update({ defaultSampling: withSampling(settings.defaultSampling, 'topK', Math.round(v)) })} />
            <NumberSetting label="Min P" value={settings.defaultSampling.minP} step={0.01} onChange={(v) => void update({ defaultSampling: withSampling(settings.defaultSampling, 'minP', v) })} />
            <NumberSetting label="Repeat penalty" value={settings.defaultSampling.repeatPenalty} step={0.01} onChange={(v) => void update({ defaultSampling: withSampling(settings.defaultSampling, 'repeatPenalty', v) })} />
          </div>
          <Field inline label="Expand reasoning blocks by default">
            <Switch checked={settings.expandThinking} onCheckedChange={(v) => void update({ expandThinking: v })} />
          </Field>
        </Card>

        <Card title="Appearance">
          <Field inline label="Theme">
            <Select className="w-36" value={settings.theme} onChange={(e) => void update({ theme: e.target.value as typeof settings.theme })}>
              <option value="dark">Dark</option>
              <option value="light">Light</option>
              <option value="system">System</option>
            </Select>
          </Field>
        </Card>

        <Card title="Keyboard shortcuts">
          <div className="grid grid-cols-[180px_1fr] gap-y-2 text-[13px]">
            {SHORTCUTS.map(([k, v]) => (
              <div key={k} className="contents">
                <div>
                  <Kbd>{k}</Kbd>
                </div>
                <div className="text-muted">{v}</div>
              </div>
            ))}
          </div>
        </Card>

        <Card title="About">
          <div className="space-y-1 text-xs text-muted">
            <div>RulZ LLM Suite {info?.version}</div>
            <div className="flex items-center gap-2">
              Data folder: <span className="font-mono">{info?.userData}</span>
              {info && (
                <Button size="sm" variant="ghost" onClick={() => void invoke('shell:openPath', info.userData)}>
                  <FolderOpen size={12} /> Open
                </Button>
              )}
            </div>
          </div>
        </Card>
      </div>
    </div>
  )
}

function withSampling<K extends keyof SamplingParams>(s: SamplingParams, key: K, value: SamplingParams[K]): SamplingParams {
  return { ...s, [key]: value }
}

function NumberSetting({ label, value, step, onChange }: { label: string; value: number; step: number; onChange: (v: number) => void }) {
  const [draft, setDraft] = useState(String(value))
  useEffect(() => {
    setDraft(String(value))
  }, [value])
  return (
    <Field label={label}>
      <Input
        type="number"
        step={step}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => {
          const n = Number(draft)
          if (draft !== '' && Number.isFinite(n) && n !== value) onChange(n)
          else setDraft(String(value))
        }}
      />
    </Field>
  )
}
