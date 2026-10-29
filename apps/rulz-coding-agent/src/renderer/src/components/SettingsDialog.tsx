import { ChevronDown, Eye, EyeOff, FolderOpen, RefreshCw, Smartphone } from 'lucide-react'
import { useEffect, useState } from 'react'
import { MAX_INSTRUCTION_CHARS } from '@shared/defaults'
import type { RemotePairing, RemoteStatus, Theme, ToolServersInfo } from '@shared/types'
import { invoke, on } from '../lib/api'
import { cn, relativeTime } from '../lib/format'
import { activeModelId, useApp } from '../stores/app'
import { ApprovalChooser, ModelSelect } from './common'
import { ModelSettings } from './ModelSettings'
import { Button, Dialog, Field, Input, NumberInput, Switch, Textarea } from './ui'

/** The tool servers from RulZ LLM Suite's mcp.json, and whether they are running. */
function ExtraTools() {
  const [info, setInfo] = useState<ToolServersInfo | null>(null)
  const [starting, setStarting] = useState(false)
  useEffect(() => {
    void invoke('tools:status').then(setInfo, () => undefined)
    return on('tools:changed', setInfo)
  }, [])
  if (!info) return <p className="text-[13px] text-faint">Loading…</p>
  if (info.error) return <p className="text-[13px] text-danger">{info.error}</p>
  if (!info.configured.length) {
    return (
      <p className="text-[13px] text-muted">
        None yet. Tool servers you set up on RulZ LLM Suite's MCP page (like REA) appear here, and I can use them in every project.
      </p>
    )
  }
  const word = (name: string): string => {
    const s = info.servers.find((x) => x.name === name)
    if (!s) return 'starts when I first need it'
    if (s.state === 'starting') return 'starting…'
    if (s.state === 'connected') return `running, ${s.tools} tools`
    return `couldn't start: ${s.error ?? 'unknown problem'}`
  }
  const allRunning = info.configured.every((n) => info.servers.find((s) => s.name === n)?.state === 'connected')
  return (
    <div className="space-y-2">
      <ul className="space-y-1 text-[13px]">
        {info.configured.map((n) => (
          <li key={n} data-testid="tool-server">
            <b>{n}</b> <span className={cn('text-muted', info.servers.find((s) => s.name === n)?.state === 'error' && 'text-danger')}>— {word(n)}</span>
          </li>
        ))}
      </ul>
      {!allRunning && (
        <Button
          size="sm"
          disabled={starting}
          onClick={async () => {
            setStarting(true)
            try {
              setInfo(await invoke('tools:start'))
            } finally {
              setStarting(false)
            }
          }}
        >
          {starting ? 'Starting…' : 'Start now'}
        </Button>
      )}
    </div>
  )
}

/** Phone access over Tailscale: the switch, the QR code to pair the phone app, and the phones connected now. */
function PhoneAccess() {
  const [status, setStatus] = useState<RemoteStatus | null>(null)
  const [pairing, setPairing] = useState<RemotePairing | null>(null)
  const [showCode, setShowCode] = useState(false)
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    void invoke('remote:status').then(setStatus, () => undefined)
    return on('remote:changed', setStatus)
  }, [])
  const ready = status?.state === 'on'
  const where = status?.addresses.join()
  useEffect(() => {
    if (ready) void invoke('remote:pairing').then(setPairing, () => undefined)
  }, [ready, where])
  if (!status) return <p className="text-[13px] text-faint">Loading…</p>

  const toggle = async (enabled: boolean): Promise<void> => {
    setBusy(true)
    try {
      setStatus(await invoke('remote:setEnabled', enabled))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-3">
      <Field label="Let my phone connect" hint="Only your own devices on Tailscale can reach it, and only with the pairing code" inline>
        <Switch checked={status.enabled} disabled={busy} onCheckedChange={(v) => void toggle(v)} label="Let my phone connect" />
      </Field>
      {status.enabled && status.state === 'starting' && <p className="text-[13px] text-muted">Starting…</p>}
      {status.enabled && status.state === 'no-tailscale' && (
        <p className="text-[13px] text-warn">Tailscale isn't connected on this computer. Start Tailscale and sign in; I'll keep checking.</p>
      )}
      {status.enabled && status.state === 'error' && <p className="text-[13px] text-danger">{status.error}</p>}
      {ready && pairing && (
        <div className="flex gap-4 rounded-lg border border-border bg-panel-2/50 p-3">
          <img
            src={`data:image/svg+xml;utf8,${encodeURIComponent(pairing.qrSvg)}`}
            alt="QR code for pairing the phone app"
            className="size-36 shrink-0 rounded-md bg-white p-1"
          />
          <div className="min-w-0 space-y-1.5 text-[13px]">
            <p className="text-muted">
              Scan this with your phone's camera, or in the RulZ Coding Agent app tap <b className="text-text">Add computer</b> and type:
            </p>
            <p>
              Address: <span className="font-mono">{pairing.host}</span>
              {status.dnsName && <span className="text-faint"> ({status.dnsName})</span>}
            </p>
            <p>
              Port: <span className="font-mono">{pairing.port}</span>
            </p>
            <p className="flex items-center gap-1.5">
              Pairing code: <span className="min-w-0 truncate font-mono">{showCode ? pairing.token : '•'.repeat(12)}</span>
              <button className="text-muted hover:text-text" aria-label={showCode ? 'Hide the code' : 'Show the code'} onClick={() => setShowCode(!showCode)}>
                {showCode ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </p>
            <Button size="sm" onClick={async () => setPairing(await invoke('remote:newToken'))} title="Phones using the old code are disconnected">
              <RefreshCw size={12} /> Make a new code
            </Button>
          </div>
        </div>
      )}
      {ready && (
        <div className="text-[13px]">
          {status.clients.length === 0 ? (
            <p className="text-faint">No phone connected right now.</p>
          ) : (
            <ul className="space-y-1">
              {status.clients.map((c) => (
                <li key={c.id} className="flex items-center gap-2">
                  <Smartphone size={14} className="text-ok" /> {c.device}
                  <span className="text-faint">
                    {c.address}, connected {relativeTime(c.since)}
                  </span>
                </li>
              ))}
            </ul>
          )}
          <p className="mt-2 text-[11px] text-faint">
            The first time, Windows may ask whether RulZ Coding Agent may use your networks. Allow it, or your phone can't connect.
          </p>
        </div>
      )}
    </div>
  )
}

/** The fixed part of the system prompt, read-only, so the owner can see what their instructions add to. */
function BuiltInPrompt() {
  const [text, setText] = useState('')
  useEffect(() => {
    void invoke('settings:builtInPrompt').then(setText, () => setText('(could not load)'))
  }, [])
  return (
    <pre className="mt-2 max-h-64 overflow-auto rounded-md border border-border bg-code p-3 font-mono text-[11.5px] whitespace-pre-wrap text-muted">
      {text || 'Loading…'}
    </pre>
  )
}

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
  const [builtIn, setBuiltIn] = useState(false)
  const [modelSettings, setModelSettings] = useState(false)
  if (!settings) return null
  const modelId = activeModelId(settings, connection)

  return (
    <Dialog open={open} onOpenChange={setOpen} title="Settings" className="w-[min(620px,calc(100vw-32px))]">
      <div className="space-y-6 px-5 py-5">
        <Field label="AI model" hint="Models marked “ready” start right away">
          {connection.models.length ? (
            <>
              <ModelSelect models={connection.models} value={modelId} onChange={(model) => void update({ model })} />
              {modelId && (
                <>
                  <button className="mt-1 flex items-center gap-1 text-[12px] text-muted hover:text-text" onClick={() => setModelSettings(!modelSettings)}>
                    <ChevronDown size={13} className={cn('transition-transform', !modelSettings && '-rotate-90')} />
                    Model settings
                  </button>
                  {modelSettings && <ModelSettings model={modelId} />}
                </>
              )}
            </>
          ) : (
            <p className="text-[13px] text-muted">Connect to RulZ LLM Suite to see its models.</p>
          )}
        </Field>

        <Field label="Keep thinking open after the answer" hint="My thought process stays open once I start answering. You can still fold it away." inline>
          <Switch checked={settings.keepThinkingOpen} onCheckedChange={(v) => void update({ keepThinkingOpen: v })} label="Keep thinking open after the answer" />
        </Field>

        <Field label="When should I ask you first?">
          <ApprovalChooser value={settings.approval} onChange={(approval) => void update({ approval })} />
        </Field>

        <Field
          label="Instructions for every project"
          hint="Added to my built-in instructions in all your projects. For one project only, use Notes in that project."
        >
          <Textarea
            rows={4}
            maxLength={MAX_INSTRUCTION_CHARS}
            aria-label="Instructions for every project"
            placeholder={'For example:\nAlways use big, easy-to-read text.\nPrefer Python apps over websites.\nExplain things briefly.'}
            defaultValue={settings.instructions}
            onBlur={(e) => e.target.value.trim() !== settings.instructions && void update({ instructions: e.target.value })}
          />
          <button className="mt-1 flex items-center gap-1 text-[12px] text-muted hover:text-text" onClick={() => setBuiltIn(!builtIn)}>
            <ChevronDown size={13} className={cn('transition-transform', !builtIn && '-rotate-90')} />
            Show my built-in instructions
          </button>
          {builtIn && <BuiltInPrompt />}
        </Field>

        <Field label="Extra tools" hint="Tool servers set up in RulZ LLM Suite, such as REA">
          <ExtraTools />
        </Field>

        <Field label="Phone access" hint="Watch and control your projects from the RulZ Coding Agent Android app">
          <PhoneAccess />
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
