import { ArrowLeft, ChevronDown, FolderOpen } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { MAX_INSTRUCTION_CHARS } from '@shared/defaults'
import type { Theme, ToolServersInfo } from '@shared/types'
import { ApprovalChooser, ModelSelect } from '../components/common'
import { FolderPicker } from '../components/FolderPicker'
import { ModelSettings } from '../components/ModelSettings'
import { Button, Field, IconButton, Input, Switch, Textarea, TopBar } from '../components/ui'
import { invoke, on } from '../lib/api'
import { cn, relativeTime } from '../lib/format'
import { isNative } from '../lib/native'
import { useAlerts } from '../stores/alerts'
import { activeModelId, useApp } from '../stores/app'
import { useLink } from '../stores/link'
import { useUi } from '../stores/ui'

const THEMES: { value: Theme; label: string }[] = [
  { value: 'dark', label: 'Dark' },
  { value: 'light', label: 'Light' },
  { value: 'system', label: 'Same as phone' }
]

function Section({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <section className="space-y-5 rounded-2xl border border-border bg-panel p-4">
      <div>
        <h2 className="text-[13px] font-medium tracking-wide text-faint uppercase">{title}</h2>
        {hint && <p className="mt-0.5 text-[12px] text-faint">{hint}</p>}
      </div>
      {children}
    </section>
  )
}

function ExtraTools() {
  const [info, setInfo] = useState<ToolServersInfo | null>(null)
  const [starting, setStarting] = useState(false)
  useEffect(() => {
    void invoke('tools:status').then(setInfo, () => undefined)
    return on('tools:changed', setInfo)
  }, [])
  if (!info) return <p className="text-[14px] text-faint">Loading…</p>
  if (info.error) return <p className="text-[14px] text-danger">{info.error}</p>
  if (!info.configured.length) return <p className="text-[14px] text-muted">None yet. Tool servers set up in RulZ LLM Suite on the computer (like REA) appear here.</p>
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
      <ul className="space-y-1 text-[14px]">
        {info.configured.map((n) => (
          <li key={n}>
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

function Notifications() {
  const prefs = useAlerts((s) => s.prefs)
  const permission = useAlerts((s) => s.permission)
  const unrestricted = useAlerts((s) => s.unrestricted)
  const backgroundError = useAlerts((s) => s.backgroundError)
  const setPrefs = useAlerts((s) => s.setPrefs)
  const test = useAlerts((s) => s.test)
  const openNotificationSettings = useAlerts((s) => s.openNotificationSettings)
  const openBatterySettings = useAlerts((s) => s.openBatterySettings)
  const refreshStatus = useAlerts((s) => s.refreshStatus)
  const problem = useAlerts((s) => s.problem)
  const last = useAlerts((s) => s.last)
  useEffect(() => void refreshStatus(), [refreshStatus])
  const any = prefs.finished || prefs.waiting

  return (
    <Section title="Notifications" hint="When this app isn't on screen. While it's open, you see everything here anyway.">
      <Field label="When a task is finished" hint="With what I said at the end" inline>
        <Switch checked={prefs.finished} onCheckedChange={(finished) => void setPrefs({ finished })} label="When a task is finished" />
      </Field>
      <Field label="When I'm waiting for you" hint="Permission requests, questions and pauses" inline>
        <Switch checked={prefs.waiting} onCheckedChange={(waiting) => void setPrefs({ waiting })} label="When I'm waiting for you" />
      </Field>
      {isNative && (
        <Field label="Stay connected in the background" hint="Needed for notifications while the app is closed. Android shows a small ongoing notification for it." inline>
          <Switch checked={prefs.background} disabled={!any} onCheckedChange={(background) => void setPrefs({ background })} label="Stay connected in the background" />
        </Field>
      )}
      {any && (permission === 'denied' || problem) && (
        <div className="space-y-2 rounded-lg border border-warn/40 bg-warn/10 p-3 text-[14px] text-warn">
          <p>{problem || "Notifications are turned off for RulZ Agent in Android's settings."}</p>
          {isNative && (
            <Button size="sm" onClick={openNotificationSettings}>
              Open notification settings
            </Button>
          )}
        </div>
      )}
      {any && isNative && !prefs.background && <p className="text-[13px] text-warn">Without the background connection, notifications only come while the app is open or was just closed.</p>}
      {backgroundError && <p className="text-[13px] text-danger">{backgroundError}</p>}
      {any && isNative && prefs.background && unrestricted === false && (
        <div className="space-y-2 text-[13px] text-muted">
          <p>If notifications stop after the phone has been locked a while, let RulZ Agent run without battery limits.</p>
          <Button size="sm" onClick={openBatterySettings}>
            Battery settings
          </Button>
        </div>
      )}
      {any && (
        <Button size="sm" onClick={() => void test()}>
          Send a test notification
        </Button>
      )}
      {last && (
        <p className={cn('text-[12px]', last.error ? 'text-danger' : 'text-faint')}>
          Last notification: “{last.title}”, {relativeTime(last.at)}
          {last.error ? `, didn't show: ${last.error}` : '.'}
        </p>
      )}
    </Section>
  )
}

export function Settings() {
  const settings = useApp((s) => s.settings)!
  const info = useApp((s) => s.info)
  const connection = useApp((s) => s.connection)
  const update = useApp((s) => s.updateSettings)
  const theme = useApp((s) => s.theme)
  const setTheme = useApp((s) => s.setTheme)
  const back = useUi((s) => s.back)
  const computer = useLink((s) => s.computers.find((c) => c.id === s.currentId))
  const device = useLink((s) => s.device)
  const setDevice = useLink((s) => s.setDevice)
  const select = useLink((s) => s.select)
  const remove = useLink((s) => s.remove)
  const [details, setDetails] = useState(false)
  const [picker, setPicker] = useState(false)
  const modelSettingsOpen = useUi((s) => s.modelSettingsOpen)
  const setModelSettingsOpen = useUi((s) => s.setModelSettingsOpen)
  const modelId = activeModelId(settings, connection)

  return (
    <div className="flex h-full flex-col">
      <TopBar>
        <IconButton label="Back" onClick={back}>
          <ArrowLeft size={21} />
        </IconButton>
        <h1 className="flex-1 px-1 text-[17px] font-semibold">Settings</h1>
      </TopBar>
      <div className="px-safe pb-safe min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-lg space-y-5 px-4 py-5">
          <Section title={`On ${computer?.name ?? 'the computer'}`} hint="These change RulZ Coding Agent on the computer too.">
            <Field label="AI model" hint="Models marked “ready” start right away">
              {connection.models.length ? (
                <>
                  <ModelSelect models={connection.models} value={modelId} onChange={(model) => void update({ model })} />
                  {modelId && (
                    <>
                      <button className="flex items-center gap-1 py-1 text-[14px] text-muted" onClick={() => setModelSettingsOpen(!modelSettingsOpen)}>
                        <ChevronDown size={16} className={cn('transition-transform', !modelSettingsOpen && '-rotate-90')} /> Model settings
                      </button>
                      {modelSettingsOpen && <ModelSettings model={modelId} />}
                    </>
                  )}
                </>
              ) : (
                <p className="text-[14px] text-muted">The computer isn't connected to RulZ LLM Suite, so there are no models to choose from.</p>
              )}
            </Field>
            {settings.keepThinkingOpen !== undefined && (
              <Field label="Keep thinking open after the answer" hint="My thought process stays open once I start answering. You can still fold it away." inline>
                <Switch checked={settings.keepThinkingOpen} onCheckedChange={(v) => void update({ keepThinkingOpen: v })} label="Keep thinking open after the answer" />
              </Field>
            )}
            <Field label="When should I ask you first?">
              <ApprovalChooser value={settings.approval} onChange={(approval) => void update({ approval })} />
            </Field>
            <Field label="Instructions for every project" hint="Added to my built-in instructions in all your projects">
              <Textarea
                rows={4}
                maxLength={MAX_INSTRUCTION_CHARS}
                aria-label="Instructions for every project"
                placeholder={'For example:\nAlways use big, easy-to-read text.\nPrefer Python apps over websites.'}
                defaultValue={settings.instructions}
                onBlur={(e) => e.target.value.trim() !== settings.instructions && void update({ instructions: e.target.value })}
              />
            </Field>
            <Field label="Extra tools" hint="Tool servers set up in RulZ LLM Suite, such as REA">
              <ExtraTools />
            </Field>
            <div>
              <button className="flex items-center gap-1 py-1 text-[14px] text-muted" onClick={() => setDetails(!details)}>
                <ChevronDown size={16} className={cn('transition-transform', !details && '-rotate-90')} /> Show details
              </button>
              {details && (
                <div className="mt-3 space-y-4">
                  <Field label="New projects go in">
                    <div className="flex gap-2">
                      <Input value={settings.projectsDir} readOnly />
                      <Button onClick={() => setPicker(true)} aria-label="Choose">
                        <FolderOpen size={16} />
                      </Button>
                    </div>
                  </Field>
                  <Field label="Most steps per task" hint="I pause and ask after this many" inline>
                    <Input
                      type="number"
                      inputMode="numeric"
                      className="w-24 text-right"
                      defaultValue={settings.maxSteps}
                      onBlur={(e) => {
                        const v = Math.round(Number(e.target.value))
                        if (v >= 5 && v <= 500 && v !== settings.maxSteps) void update({ maxSteps: v })
                      }}
                    />
                  </Field>
                  <Field label="Time limit for one command (minutes)" inline>
                    <Input
                      type="number"
                      inputMode="numeric"
                      className="w-24 text-right"
                      defaultValue={settings.commandTimeoutMin}
                      onBlur={(e) => {
                        const v = Number(e.target.value)
                        if (v >= 1 && v <= 120 && v !== settings.commandTimeoutMin) void update({ commandTimeoutMin: v })
                      }}
                    />
                  </Field>
                  <p className="text-[12px] text-faint">
                    RulZ LLM Suite address and API key can only be changed on the computer. RulZ Coding Agent {info?.version} on the computer.
                  </p>
                </div>
              )}
            </div>
          </Section>

          <Notifications />

          <Section title="This phone">
            <Field label="Look">
              <div className="flex gap-1.5">
                {THEMES.map((t) => (
                  <Button key={t.value} size="sm" variant={theme === t.value ? 'primary' : 'secondary'} onClick={() => void setTheme(t.value)}>
                    {t.label}
                  </Button>
                ))}
              </div>
            </Field>
            <Field label="This phone's name" hint="Shown on the computer when this phone connects">
              <Input defaultValue={device} onBlur={(e) => e.target.value.trim() !== device && void setDevice(e.target.value)} />
            </Field>
            <Field label="Computer" hint={computer ? `${computer.host}:${computer.port}` : undefined}>
              <div className="flex flex-wrap gap-2">
                <Button onClick={() => select(null)}>Switch computer</Button>
                {computer && (
                  <Button variant="ghost" className="text-danger" onClick={() => void remove(computer.id)}>
                    Forget this computer
                  </Button>
                )}
              </div>
            </Field>
            <p className="text-[12px] text-faint">RulZ Coding Agent for Android {__APP_VERSION__}</p>
          </Section>
        </div>
      </div>
      <FolderPicker
        open={picker}
        onOpenChange={setPicker}
        title="Where should new projects go?"
        start={settings.projectsDir}
        action="Use this folder"
        onPick={(projectsDir) => {
          void update({ projectsDir })
          setPicker(false)
        }}
      />
    </div>
  )
}
