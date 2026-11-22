import { ArrowLeft } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import type { Theme } from '@shared/types'
import { Button, Field, IconButton, Input, Select, Switch, TopBar } from '../components/ui'
import { cn, relativeTime } from '../lib/format'
import { isNative } from '../lib/native'
import { link } from '../lib/remote'
import { useAlerts } from '../stores/alerts'
import { useApp } from '../stores/app'
import { useLink } from '../stores/link'
import { useProject } from '../stores/project'
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

/** The model the open project writes with (the same setting as in the Writing page on the computer). */
function ProjectModel() {
  const project = useProject((s) => s.state?.project)
  const setModel = useProject((s) => s.setModel)
  const [models, setModels] = useState<{ id: string; name: string; loaded: boolean }[] | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    if (!project) return
    let live = true
    link.invoke('writing:models').then(
      (m) => live && setModels(m),
      (e) => live && setError(e instanceof Error ? e.message : String(e))
    )
    return () => {
      live = false
    }
  }, [project])

  if (!project) return <p className="text-[14px] text-muted">Open a project to choose the model it writes with.</p>
  if (error) return <p className="text-[14px] text-danger">{error}</p>
  if (!models) return <p className="text-[14px] text-faint">Loading…</p>
  const sorted = [...models].sort((a, b) => Number(b.loaded) - Number(a.loaded) || a.name.localeCompare(b.name))
  return (
    <Field label={`Model for “${project.title}”`} hint="Models marked “ready” start right away. The others are loaded by the computer first.">
      <Select value={project.modelId ?? ''} onChange={(e) => void setModel(e.target.value || null)}>
        <option value="">The model that is active on the computer</option>
        {project.modelId && !models.some((m) => m.id === project.modelId) && <option value={project.modelId}>{project.modelId}</option>}
        {sorted.map((m) => (
          <option key={m.id} value={m.id}>
            {m.name}
            {m.loaded ? ' (ready)' : ''}
          </option>
        ))}
      </Select>
    </Field>
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

  return (
    <Section title="Notifications" hint="When this app isn't on screen. While it's open, you see everything here anyway.">
      <Field label="When a writing job finishes" hint="With the start of what was written" inline>
        <Switch checked={prefs.finished} onCheckedChange={(finished) => void setPrefs({ finished })} label="When a writing job finishes" />
      </Field>
      {isNative && (
        <Field label="Stay connected in the background" hint="Needed for notifications while the app is closed. Android shows a small ongoing notification for it." inline>
          <Switch checked={prefs.background} disabled={!prefs.finished} onCheckedChange={(background) => void setPrefs({ background })} label="Stay connected in the background" />
        </Field>
      )}
      {prefs.finished && (permission === 'denied' || problem) && (
        <div className="space-y-2 rounded-lg border border-warn/40 bg-warn/10 p-3 text-[14px] text-warn">
          <p>{problem || "Notifications are turned off for RulZ Writer in Android's settings."}</p>
          {isNative && (
            <Button size="sm" onClick={openNotificationSettings}>
              Open notification settings
            </Button>
          )}
        </div>
      )}
      {prefs.finished && isNative && !prefs.background && <p className="text-[13px] text-warn">Without the background connection, notifications only come while the app is open or was just closed.</p>}
      {backgroundError && <p className="text-[13px] text-danger">{backgroundError}</p>}
      {prefs.finished && isNative && prefs.background && unrestricted === false && (
        <div className="space-y-2 text-[13px] text-muted">
          <p>If notifications stop after the phone has been locked a while, let RulZ Writer run without battery limits.</p>
          <Button size="sm" onClick={openBatterySettings}>
            Battery settings
          </Button>
        </div>
      )}
      {prefs.finished && (
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
  const theme = useApp((s) => s.theme)
  const setTheme = useApp((s) => s.setTheme)
  const back = useUi((s) => s.back)
  const computers = useLink((s) => s.computers)
  const currentId = useLink((s) => s.currentId)
  const computer = computers.find((c) => c.id === currentId)
  const device = useLink((s) => s.device)
  const setDevice = useLink((s) => s.setDevice)
  const select = useLink((s) => s.select)
  const remove = useLink((s) => s.remove)

  return (
    <div className="flex h-full flex-col">
      <TopBar>
        <IconButton label="Back" onClick={back}>
          <ArrowLeft size={21} />
        </IconButton>
        <h1 className="flex-1 px-1 font-[family-name:var(--font-display)] text-[17px] font-semibold">Settings</h1>
      </TopBar>
      <div className="px-safe pb-safe min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-lg space-y-5 px-4 py-5">
          <Section title={`On ${computer?.name ?? 'the computer'}`} hint="These change RulZ LLM Suite on the computer too.">
            <ProjectModel />
          </Section>

          <Notifications />

          <Section title="This phone">
            <Field label="Look">
              <div className="flex flex-wrap gap-1.5">
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
            <p className="text-[12px] text-faint">RulZ Writer for Android {__APP_VERSION__}</p>
          </Section>
        </div>
      </div>
    </div>
  )
}
